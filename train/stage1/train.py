"""Unified training script. Usage: python train.py --exp a"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import (
    TRAIN_CFG, EXP_CONFIGS,
    build_voxtell_from_checkpoint,
    apply_for_exp, count_params,
)

# Reuse dataset from stage1_workspace (base) with local override for neg sampling
sys.path.insert(0, str(THIS_DIR))  # local dataset takes priority
STAGE1_WORKSPACE = THIS_DIR.parent / "stage1_workspace"
sys.path.append(str(STAGE1_WORKSPACE))
from dataset.rex_stage1_dataset import ReXStage1Dataset  # noqa: E402


# ======================================================================
# Loss & metrics (same as stage1_workspace)
# ======================================================================

class CombinedLoss(nn.Module):
    def __init__(self, dice_w=1.0, bce_w=1.0):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        p = torch.sigmoid(pred)
        inter = (p * target).sum()
        union = p.sum() + target.sum()
        dice = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)
        return self.dice_w * dice + self.bce_w * self.bce(pred, target)


@torch.no_grad()
def compute_dice_recall(pred, target, thr=0.5):
    pb = (torch.sigmoid(pred) > thr).float()
    inter = (pb * target).sum().item()
    ps = pb.sum().item()
    ts = target.sum().item()
    d = (2 * inter + 1e-6) / (ps + ts + 1e-6)
    r = (inter + 1e-6) / (ts + 1e-6)
    return d, r


# ======================================================================
# Training
# ======================================================================

def write_progress(path: Path, payload: dict):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def train_epoch(model, loader, opt, sched, loss_fn, scaler, device, grad_accum, epoch, max_epochs, progress_path):
    model.train()
    model.encoder.eval()
    total_loss = 0.0
    total_dice, total_recall, n = 0.0, 0.0, 0
    opt.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        msk = batch["mask"].to(device, non_blocking=True)
        txt = batch["text_embedding"].to(device, non_blocking=True).unsqueeze(1)

        with autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            pred = model(img, txt)
            if isinstance(pred, list):
                pred = pred[0]
            loss = loss_fn(pred, msk.unsqueeze(0)) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()

        d, r = compute_dice_recall(pred.detach().float(), msk.unsqueeze(0))
        total_loss += loss.item() * grad_accum
        total_dice += d
        total_recall += r
        n += 1

        write_progress(progress_path, {
            "stage": "train",
            "epoch": epoch,
            "max_epochs": max_epochs,
            "step": step + 1,
            "steps": len(loader),
            "epoch_progress": (step + 1) / max(len(loader), 1),
            "loss_avg": total_loss / max(n, 1),
            "dice_avg": total_dice / max(n, 1),
            "recall_avg": total_recall / max(n, 1),
            "elapsed_seconds": round(time.time() - t0, 1),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return total_loss / max(n, 1), total_dice / max(n, 1), total_recall / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, max_epochs, progress_path):
    model.eval()
    total_loss = 0.0
    total_dice, total_recall, n = 0.0, 0.0, 0
    t0 = time.time()
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        msk = batch["mask"].to(device, non_blocking=True)
        txt = batch["text_embedding"].to(device, non_blocking=True).unsqueeze(1)
        with autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            pred = model(img, txt)
            if isinstance(pred, list):
                pred = pred[0]
            loss = loss_fn(pred, msk.unsqueeze(0))
        d, r = compute_dice_recall(pred.float(), msk.unsqueeze(0))
        total_loss += loss.item()
        total_dice += d
        total_recall += r
        n += 1
        write_progress(progress_path, {
            "stage": "val",
            "epoch": epoch,
            "max_epochs": max_epochs,
            "step": n,
            "steps": len(loader),
            "epoch_progress": n / max(len(loader), 1),
            "loss_avg": total_loss / max(n, 1),
            "dice_avg": total_dice / max(n, 1),
            "recall_avg": total_recall / max(n, 1),
            "elapsed_seconds": round(time.time() - t0, 1),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return total_loss / max(n, 1), total_dice / max(n, 1), total_recall / max(n, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=list(EXP_CONFIGS))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--manifest-train", type=Path, required=True)
    parser.add_argument("--manifest-val", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs_fixed"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=None, help="Override config max_epochs for smoke tests.")
    parser.add_argument("--val-interval", type=int, default=None, help="Override config val_interval.")
    parser.add_argument("--limit-train", type=int, default=None, help="Use only the first N train samples.")
    parser.add_argument("--limit-val", type=int, default=None, help="Use only the first N val samples.")
    parser.add_argument("--progress-json", type=Path, default=None,
                        help="Progress JSON path. Defaults to output_root/exp_x/progress.json.")
    args = parser.parse_args()

    cfg = dict(TRAIN_CFG)
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.val_interval is not None:
        cfg["val_interval"] = args.val_interval
    exp_cfg = EXP_CONFIGS[args.exp]
    lr = exp_cfg.get("lr", cfg["lr"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = args.output_root / f"exp_{args.exp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    progress_path = args.progress_json or (out_dir / "progress.json")
    metrics_path = out_dir / "metrics.jsonl"

    print(f"Experiment: {args.exp}")
    print(f"Config: r={exp_cfg['r']}, lr={lr:g}, unfreeze_enc={exp_cfg['lr_unfreeze_enc']}")
    write_progress(progress_path, {
        "stage": "init",
        "exp": args.exp,
        "max_epochs": cfg["max_epochs"],
        "val_interval": cfg["val_interval"],
        "num_workers": args.num_workers,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Datasets
    train_ds = ReXStage1Dataset(args.manifest_train, args.image_root, args.label_root,
                                args.embedding_dir, cfg["patch_size"], prob_foreground=cfg["prob_fg"],
                                neg_ratio=cfg["neg_ratio"])
    val_ds = ReXStage1Dataset(args.manifest_val, args.image_root, args.label_root,
                              args.embedding_dir, cfg["patch_size"], prob_foreground=cfg["prob_fg"],
                              neg_ratio=0)  # val: positive only, for clean metrics
    if args.limit_train is not None:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val is not None:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    write_progress(progress_path, {
        "stage": "data_ready",
        "exp": args.exp,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "max_epochs": cfg["max_epochs"],
        "val_interval": cfg["val_interval"],
        "num_workers": args.num_workers,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Model (CPU → DoRA → GPU)
    r_val = exp_cfg["r"] if exp_cfg["r"] > 0 else 8  # exp A uses 0 but needs a value for apply
    print(f"Loading VoxTell on CPU, applying config for exp {args.exp}...")
    model = build_voxtell_from_checkpoint(args.model_dir, device=torch.device("cpu"))
    model = apply_for_exp(model, args.exp, r=r_val, alpha=r_val * 2)
    model = model.to(device)
    count_params(model)
    write_progress(progress_path, {
        "stage": "model_ready",
        "exp": args.exp,
        "device": str(device),
        "max_epochs": cfg["max_epochs"],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Optimizer
    if exp_cfg["lr_unfreeze_enc"]:
        enc_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
        enc_param_ids = {id(p) for p in enc_params}
        other_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in enc_param_ids
        ]
        optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": cfg["lr_encoder"]},
            {"params": other_params, "lr": lr},
        ], weight_decay=cfg["weight_decay"])
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg["weight_decay"])

    # Scheduler
    total_steps = len(train_loader) * cfg["max_epochs"] // cfg["grad_accum"]
    warmup_steps = len(train_loader) * cfg["warmup_epochs"] // cfg["grad_accum"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    loss_fn = CombinedLoss(cfg["dice_weight"], cfg["bce_weight"])

    best_val_dice = -1.0
    patience = 0

    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        epoch_num = epoch + 1
        write_progress(progress_path, {
            "stage": "epoch_start",
            "exp": args.exp,
            "epoch": epoch_num,
            "max_epochs": cfg["max_epochs"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        tl, td, tr = train_epoch(model, train_loader, optimizer, scheduler, loss_fn,
                                 scaler, device, cfg["grad_accum"], epoch_num, cfg["max_epochs"], progress_path)
        epoch_seconds = time.time() - t0
        print(f"[E{epoch_num:02d}] train loss={tl:.4f} dice={td:.4f} recall={tr:.4f} time={epoch_seconds:.0f}s")
        write_progress(progress_path, {
            "stage": "epoch_train_done",
            "exp": args.exp,
            "epoch": epoch_num,
            "max_epochs": cfg["max_epochs"],
            "train_loss": tl,
            "train_dice": td,
            "train_recall": tr,
            "train_seconds": round(epoch_seconds, 1),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        append_jsonl(metrics_path, {
            "stage": "train",
            "exp": args.exp,
            "epoch": epoch_num,
            "max_epochs": cfg["max_epochs"],
            "loss": tl,
            "dice": td,
            "recall": tr,
            "seconds": round(epoch_seconds, 1),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        if epoch_num % cfg["val_interval"] == 0 or epoch == cfg["max_epochs"] - 1:
            vl, vd, vr = validate(model, val_loader, loss_fn, device, epoch_num, cfg["max_epochs"], progress_path)
            print(f"  val loss={vl:.4f} dice={vd:.4f} recall={vr:.4f}")
            append_jsonl(metrics_path, {
                "stage": "val",
                "exp": args.exp,
                "epoch": epoch_num,
                "max_epochs": cfg["max_epochs"],
                "loss": vl,
                "dice": vd,
                "recall": vr,
                "best_val_dice_before_update": best_val_dice,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

            ckpt = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_dice": best_val_dice, "val_dice": vd, "val_recall": vr,
                "exp": args.exp, "config": cfg,
            }
            ckpt_path = out_dir / "checkpoints" / "latest.pt"
            torch.save(ckpt, ckpt_path)

            if vd > best_val_dice:
                best_val_dice = vd
                patience = 0
                import shutil
                shutil.copy2(ckpt_path, out_dir / "checkpoints" / "best.pt")
                print(f"  [best] dice={vd:.4f}")
            else:
                patience += 1
                if patience >= cfg["early_stop_patience"]:
                    print(f"Early stop at epoch {epoch+1}")
                    break
            write_progress(progress_path, {
                "stage": "checkpoint_saved",
                "exp": args.exp,
                "epoch": epoch_num,
                "max_epochs": cfg["max_epochs"],
                "val_loss": vl,
                "val_dice": vd,
                "val_recall": vr,
                "best_val_dice": best_val_dice,
                "checkpoint": str(ckpt_path),
                "best_checkpoint": str(out_dir / "checkpoints" / "best.pt"),
                "metrics": str(metrics_path),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    print(f"Done. Best val dice: {best_val_dice:.4f}")
    write_progress(progress_path, {
        "stage": "done",
        "exp": args.exp,
        "max_epochs": cfg["max_epochs"],
        "best_val_dice": best_val_dice,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


if __name__ == "__main__":
    main()
