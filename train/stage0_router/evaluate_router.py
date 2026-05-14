import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from router_model import RouterHead
from router_policy import (
    CATEGORY_LABELS,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    ID_TO_TIGHTNESS,
    TIGHTNESS_LABELS,
    apply_fail_open,
    category_to_tightness,
)
from router_utils import write_csv, write_json


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def confusion_matrix(y_true, y_pred, labels):
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    rows = []
    for i, label in enumerate(labels):
        row = {"true": label}
        for j, pred_label in enumerate(labels):
            row[pred_label] = int(mat[i, j])
        rows.append(row)
    return rows


def macro_f1(y_true, y_pred, num_classes):
    f1s = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def load_model(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]["model"]
    model = RouterHead(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        num_blocks=int(cfg.get("num_blocks", 2)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_arrays(model, data, batch_size, device):
    x = data["embeddings"]
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    cat_probs = []
    tight_probs = []
    for (xb,) in loader:
        out = model(xb.to(device))
        cat_probs.append(F.softmax(out["category_logits"], dim=1).cpu())
        tight_probs.append(F.softmax(out["tightness_logits"], dim=1).cpu())
    return torch.cat(cat_probs).numpy(), torch.cat(tight_probs).numpy()


def evaluate_split(model, data, split, batch_size, thresholds, report_dir, device):
    cat_probs, tight_probs = predict_arrays(model, data, batch_size, device)
    y_cat = data["category_labels"].numpy()
    y_tight = data["tightness_labels"].numpy()
    pred_cat = cat_probs.argmax(axis=1)
    pred_tight = tight_probs.argmax(axis=1)

    final_tight = []
    post_cat = []
    fail_open_count = 0
    decisions = []
    error_cases = []
    for i, prompt in enumerate(data["prompts"]):
        decision = apply_fail_open(
            cat_probs[i].tolist(),
            tight_probs[i].tolist(),
            prompt,
            category_threshold=float(thresholds["category_threshold"]),
            tightness_threshold=float(thresholds["tightness_threshold"]),
        )
        final_tight_id = TIGHTNESS_LABELS.index(decision.final_tightness)
        post_cat_id = CATEGORY_TO_ID[decision.pred_category]
        final_tight.append(final_tight_id)
        post_cat.append(post_cat_id)
        if decision.fail_open_reason:
            fail_open_count += 1
        true_cat = ID_TO_CATEGORY[int(y_cat[i])]
        true_tight = ID_TO_TIGHTNESS[int(y_tight[i])]
        row = {
            "id": data["ids"][i],
            "prompt": prompt,
            "true_category": true_cat,
            "raw_pred_category": decision.raw_pred_category,
            "pred_category": decision.pred_category,
            "category_postprocess_reason": decision.category_postprocess_reason,
            "category_confidence": decision.category_confidence,
            "true_tightness": true_tight,
            "category_tightness": category_to_tightness(decision.pred_category),
            "pred_tightness": decision.pred_tightness,
            "tightness_confidence": decision.tightness_confidence,
            "final_tightness": decision.final_tightness,
            "laterality": decision.laterality,
            "final_policy": decision.final_policy,
            "fail_open_reason": decision.fail_open_reason,
            "dangerous_raw": int(pred_tight[i] > y_tight[i]),
            "dangerous_final": int(final_tight_id > y_tight[i]),
        }
        decisions.append(row)
        if row["true_category"] != row["pred_category"] or row["dangerous_final"]:
            error_cases.append(row)

    final_tight = np.array(final_tight)
    post_cat = np.array(post_cat)
    metrics = {
        "split": split,
        "n": int(len(y_cat)),
        "category_accuracy_raw": float(np.mean(pred_cat == y_cat)),
        "category_macro_f1_raw": macro_f1(y_cat, pred_cat, len(CATEGORY_LABELS)),
        "category_accuracy_postprocessed": float(np.mean(post_cat == y_cat)),
        "category_macro_f1_postprocessed": macro_f1(y_cat, post_cat, len(CATEGORY_LABELS)),
        "category_postprocess_rate": float(np.mean(post_cat != pred_cat)),
        "tightness_accuracy_raw": float(np.mean(pred_tight == y_tight)),
        "tightness_accuracy_final": float(np.mean(final_tight == y_tight)),
        "dangerous_error_rate_raw": float(np.mean(pred_tight > y_tight)),
        "dangerous_error_rate_final": float(np.mean(final_tight > y_tight)),
        "conservative_to_moderate_rate_final": float(
            np.mean((y_tight == 0) & (final_tight == 1))
        ),
        "conservative_to_aggressive_rate_final": float(
            np.mean((y_tight == 0) & (final_tight == 2))
        ),
        "moderate_to_aggressive_rate_final": float(
            np.mean((y_tight == 1) & (final_tight == 2))
        ),
        "fail_open_rate": float(fail_open_count / max(1, len(y_cat))),
    }

    write_json(metrics, report_dir / f"{split}_metrics.json")
    write_csv(confusion_matrix(y_cat, pred_cat, CATEGORY_LABELS), report_dir / f"{split}_category_confusion_matrix_raw.csv")
    write_csv(confusion_matrix(y_cat, post_cat, CATEGORY_LABELS), report_dir / f"{split}_category_confusion_matrix_postprocessed.csv")
    write_csv(confusion_matrix(y_tight, final_tight, TIGHTNESS_LABELS), report_dir / f"{split}_tightness_confusion_matrix.csv")
    write_csv(error_cases, report_dir / f"{split}_error_cases.csv")
    write_csv(decisions, report_dir / f"{split}_predictions.csv")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="router_training/artifacts/models/router_head_best.pt")
    parser.add_argument("--splits", nargs="+", default=["val", "holdout"])
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(checkpoint_path, device)
    artifact_dir = resolve_path(ckpt["config"]["artifact_dir"])
    batch_size = int(ckpt["config"]["train"]["batch_size"])
    thresholds = ckpt["config"]["fail_open"]
    report_dir = artifact_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for split in args.splits:
        data = torch.load(artifact_dir / "embeddings" / f"{split}.pt", map_location="cpu")
        all_metrics[split] = evaluate_split(
            model, data, split, batch_size, thresholds, report_dir, device
        )
    write_json(all_metrics, report_dir / "eval_summary.json")


if __name__ == "__main__":
    main()
