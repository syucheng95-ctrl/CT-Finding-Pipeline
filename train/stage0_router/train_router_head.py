import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from router_model import RouterHead
from router_policy import ID_TO_TIGHTNESS, TIGHTNESS_LABELS
from router_utils import write_json


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(artifact_dir: Path, split: str):
    return torch.load(artifact_dir / "embeddings" / f"{split}.pt", map_location="cpu")


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights / weights.mean()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def dangerous_error_rate(true_tightness: np.ndarray, pred_tightness: np.ndarray) -> float:
    dangerous = pred_tightness > true_tightness
    return float(np.mean(dangerous)) if len(true_tightness) else 0.0


@torch.no_grad()
def evaluate(model, data, batch_size: int, device: torch.device):
    model.eval()
    x = data["embeddings"]
    y_cat = data["category_labels"]
    y_tight = data["tightness_labels"]
    loader = DataLoader(TensorDataset(x, y_cat, y_tight), batch_size=batch_size, shuffle=False)
    cat_preds = []
    tight_preds = []
    for xb, _, _ in loader:
        out = model(xb.to(device))
        cat_preds.append(out["category_logits"].argmax(dim=1).cpu())
        tight_preds.append(out["tightness_logits"].argmax(dim=1).cpu())
    cat_pred = torch.cat(cat_preds).numpy()
    tight_pred = torch.cat(tight_preds).numpy()
    cat_true = y_cat.numpy()
    tight_true = y_tight.numpy()
    return {
        "category_accuracy": float(np.mean(cat_pred == cat_true)),
        "category_macro_f1": macro_f1(cat_true, cat_pred, 14),
        "tightness_accuracy": float(np.mean(tight_pred == tight_true)),
        "dangerous_error_rate": dangerous_error_rate(tight_true, tight_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="router_training/configs/router_qwen.json")
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    artifact_dir = resolve_path(config["artifact_dir"])
    model_cfg = config["model"]
    train_cfg = config["train"]
    set_seed(int(train_cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(artifact_dir, "train")
    val = load_split(artifact_dir, "val")
    input_dim = int(train["embeddings"].shape[1])

    model = RouterHead(
        input_dim=input_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        num_blocks=int(model_cfg.get("num_blocks", 2)),
    ).to(device)

    cat_w = class_weights(train["category_labels"], 14).to(device)
    tight_w = class_weights(train["tightness_labels"], 3).to(device)
    ce_cat = nn.CrossEntropyLoss(weight=cat_w)
    ce_tight = nn.CrossEntropyLoss(weight=tight_w)
    lambda_aux = float(model_cfg.get("lambda_aux", 0.3))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg["epochs"])
    )

    ds = TensorDataset(
        train["embeddings"],
        train["category_labels"],
        train["tightness_labels"],
    )
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    model_dir = artifact_dir / "models"
    report_dir = artifact_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best_score = -math.inf
    best_epoch = -1
    bad_epochs = 0
    history = []
    patience = int(train_cfg.get("patience", 8))

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        losses = []
        for xb, y_cat, y_tight in loader:
            xb = xb.to(device)
            y_cat = y_cat.to(device)
            y_tight = y_tight.to(device)
            out = model(xb)
            loss_cat = ce_cat(out["category_logits"], y_cat)
            loss_tight = ce_tight(out["tightness_logits"], y_tight)
            loss = loss_cat + lambda_aux * loss_tight
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        metrics = evaluate(model, val, int(train_cfg["batch_size"]), device)
        score = (
            metrics["tightness_accuracy"]
            - 2.0 * metrics["dangerous_error_rate"]
            + 0.2 * metrics["category_macro_f1"]
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_score": float(score),
            **metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        torch.save(
            {
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "config": config,
                "epoch": epoch,
                "metrics": metrics,
            },
            model_dir / "router_head_last.pt",
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "config": config,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                model_dir / "router_head_best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    write_json(
        {"best_epoch": best_epoch, "best_score": best_score, "history": history},
        report_dir / "train_history.json",
    )


if __name__ == "__main__":
    main()
