import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

mp.set_start_method("spawn", force=True)

from dataset import GROUP_ORDER, ReXStage2ROIPatchDataset
from model import create_stunet_model

# V3 defaults for 48 GB GPUs (L20/A40/A6000 class).
GROUP_PATCH_SHAPES = {
    "small": (64, 64, 32),
    "medium": (96, 96, 48),
    "large": (224, 224, 64),
    "xlarge": (256, 256, 160),
}

# Per-group batch sizes
GROUP_CONFIGS = {
    "small":  {"batch": 24},
    "medium": {"batch": 12},
    "large":  {"batch": 4},
    "xlarge": {"batch": 2},
}

# Random crop for groups that exceed patch size; center for fully-covered groups
CROP_MODES = {
    "small": "center",
    "medium": "center",
    "large": "random",
    "xlarge": "random",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def batch_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    preds = torch.argmax(logits, dim=1).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    recall = (intersection + eps) / (target_sum + eps)
    precision = (intersection + eps) / (pred_sum + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "recall": float(recall.mean().item()),
        "precision": float(precision.mean().item()),
    }


def build_train_loaders(manifest: Path, data_root: Path, steps_per_group: int, num_workers: int):
    """Build one DataLoader per group with balanced RandomSamplers."""
    loaders = {}
    for group in GROUP_ORDER:
        cfg = GROUP_CONFIGS[group]
        shape = GROUP_PATCH_SHAPES[group]
        batch = cfg["batch"]

        ds = ReXStage2ROIPatchDataset(
            manifest_path=manifest,
            data_root=data_root,
            split="train",
            target_shape=shape,
            augment=True,
            group_filter=group,
            crop_mode=CROP_MODES[group],
        )

        n_samples = steps_per_group * batch
        replacement = n_samples > len(ds)
        sampler = RandomSampler(ds, replacement=replacement, num_samples=n_samples)

        loader = DataLoader(
            ds,
            batch_size=batch,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            drop_last=False,
        )
        loaders[group] = loader

    return loaders


def build_val_loaders(manifest: Path, data_root: Path, num_workers: int):
    """Build one DataLoader per group for validation (no augmentation, no shuffle)."""
    loaders = {}
    for group in GROUP_ORDER:
        cfg = GROUP_CONFIGS[group]
        shape = GROUP_PATCH_SHAPES[group]

        ds = ReXStage2ROIPatchDataset(
            manifest_path=manifest,
            data_root=data_root,
            split="val",
            target_shape=shape,
            augment=False,
            group_filter=group,
            crop_mode="center",
        )

        if len(ds) == 0:
            loaders[group] = None
            continue

        loader = DataLoader(
            ds,
            batch_size=cfg["batch"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        loaders[group] = loader

    return loaders


def validate_manifest_files(manifest: Path, data_root: Path):
    """Fail fast on missing or empty files before expensive training starts."""
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checked = 0
    bad = []
    split_counts = {g: 0 for g in GROUP_ORDER}

    for row in rows:
        if row.get("split") not in {"train", "val"}:
            continue
        group = row.get("group")
        if group not in split_counts:
            from dataset import get_group

            group = get_group(tuple(row["roi_shape_hwd"]))
        split_counts[group] += 1

        for key in ("roi_image_path", "roi_mask_path"):
            rel = row.get(key)
            path = data_root / rel if rel else None
            checked += 1
            if path is None:
                bad.append((row.get("component_sample_id"), key, "<missing manifest field>", "missing"))
            elif not path.exists():
                bad.append((row.get("component_sample_id"), key, str(path), "missing"))
            elif path.stat().st_size == 0:
                bad.append((row.get("component_sample_id"), key, str(path), "empty"))
            if len(bad) >= 20:
                break
        if len(bad) >= 20:
            break

    empty_groups = [g for g, n in split_counts.items() if n == 0]
    if bad or empty_groups:
        details = "\n".join(
            f"  {sample_id} {key} {reason}: {path}"
            for sample_id, key, path, reason in bad
        )
        if empty_groups:
            details += "\n  empty train/val groups: " + ", ".join(empty_groups)
        raise RuntimeError(
            "Dataset preflight failed. Fix the ROI dataset before training.\n"
            f"Checked {checked} train/val files under {data_root}.\n"
            f"{details}"
        )

    print(f"Dataset preflight ok: checked {checked} train/val files; group counts={split_counts}")


def parse_shape(values: list[int], name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must have exactly 3 integers: H W D")
    return tuple(int(v) for v in values)


def apply_runtime_group_config(args):
    GROUP_PATCH_SHAPES["small"] = parse_shape(args.small_patch, "--small-patch")
    GROUP_PATCH_SHAPES["medium"] = parse_shape(args.medium_patch, "--medium-patch")
    GROUP_PATCH_SHAPES["large"] = parse_shape(args.large_patch, "--large-patch")
    GROUP_PATCH_SHAPES["xlarge"] = parse_shape(args.xlarge_patch, "--xlarge-patch")

    GROUP_CONFIGS["small"]["batch"] = int(args.small_batch)
    GROUP_CONFIGS["medium"]["batch"] = int(args.medium_batch)
    GROUP_CONFIGS["large"]["batch"] = int(args.large_batch)
    GROUP_CONFIGS["xlarge"]["batch"] = int(args.xlarge_batch)


def validate(model, val_loaders: dict, device: torch.device):
    """Evaluate all val samples per-group and return per-group + overall metrics."""
    model.eval()
    group_metrics = {}
    all_metrics = []

    for group in GROUP_ORDER:
        loader = val_loaders.get(group)
        if loader is None:
            continue

        group_dice = []
        group_recall = []
        group_precision = []
        group_losses = []

        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(x)
                    ce = F.cross_entropy(logits, y)
                    dice_l = dice_loss_from_logits(logits, y)
                    loss = ce + dice_l

                group_losses.append(float(loss.item()))

                for j in range(x.size(0)):
                    m = batch_segmentation_metrics(logits[j : j + 1], y[j : j + 1])
                    group_dice.append(m["dice"])
                    group_recall.append(m["recall"])
                    group_precision.append(m["precision"])
                    all_metrics.append(m)

        n = len(group_dice)
        group_metrics[group] = {
            "n": n,
            "loss": float(np.mean(group_losses)) if group_losses else 0.0,
            "dice": float(np.mean(group_dice)) if group_dice else 0.0,
            "recall": float(np.mean(group_recall)) if group_recall else 0.0,
            "precision": float(np.mean(group_precision)) if group_precision else 0.0,
        }

    total_n = sum(g["n"] for g in group_metrics.values())
    overall = {
        "n": total_n,
        "loss": float(np.mean([g["loss"] for g in group_metrics.values()])) if group_metrics else 0.0,
        "dice": float(np.mean([m["dice"] for m in all_metrics])) if all_metrics else 0.0,
        "recall": float(np.mean([m["recall"] for m in all_metrics])) if all_metrics else 0.0,
        "precision": float(np.mean([m["precision"] for m in all_metrics])) if all_metrics else 0.0,
    }

    return group_metrics, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifest.jsonl"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--variant", choices=["STU-Net-S", "STU-Net-B"], default="STU-Net-S")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps-per-group", type=int, default=400,
                        help="Training steps per group per epoch (total steps = 4x this)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/stage2_v3"))
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Full resume: model + optimizer + scaler + epoch")
    parser.add_argument("--load-weights", type=Path, default=None,
                        help="Load model weights only (fresh optimizer, epoch=1)")
    parser.add_argument("--small-patch", nargs=3, type=int, default=[64, 64, 32])
    parser.add_argument("--medium-patch", nargs=3, type=int, default=[96, 96, 48])
    parser.add_argument("--large-patch", nargs=3, type=int, default=[224, 224, 64])
    parser.add_argument("--xlarge-patch", nargs=3, type=int, default=[256, 256, 160])
    parser.add_argument("--small-batch", type=int, default=24)
    parser.add_argument("--medium-batch", type=int, default=12)
    parser.add_argument("--large-batch", type=int, default=4)
    parser.add_argument("--xlarge-batch", type=int, default=2)
    parser.add_argument("--scheduler-t0", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm after AMP unscale; set <=0 to disable.")
    parser.add_argument("--keep-epoch-checkpoints", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Skip manifest file existence/size preflight.")
    args = parser.parse_args()

    set_seed(args.seed)
    apply_runtime_group_config(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Multi-scale round-robin training V3")
    print(f"  steps_per_group: {args.steps_per_group}")
    print(f"  total steps/epoch: {args.steps_per_group * len(GROUP_ORDER)}")
    for g in GROUP_ORDER:
        shape = GROUP_PATCH_SHAPES[g]
        batch = GROUP_CONFIGS[g]["batch"]
        cm = CROP_MODES[g]
        print(f"  {g:>8}: patch={shape[0]}x{shape[1]}x{shape[2]}  batch={batch}  crop={cm}")
    print("=" * 60)

    if not args.skip_data_check:
        validate_manifest_files(args.manifest, args.data_root)

    # ── Model ──
    model = create_stunet_model(variant=args.variant, pretrained_dataset="TotalSegmentator", out_channels=2)
    model = model.to(device)

    # ── Optimizer & scaler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── LR scheduler ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.scheduler_t0, T_mult=1
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.save_dir / "metrics_history.json"
    history = []
    best_val_dice = -1.0
    start_epoch = 1

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            best_val_dice = max(item.get("val_dice", -1.0) for item in history)

    if args.load_weights is not None:
        ckpt = torch.load(args.load_weights, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {args.load_weights} (fresh optimizer, epoch=1)")
    elif args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        print(f"Resumed from {args.resume_from} at epoch {start_epoch} (lr reset to {args.lr})")

    # ── Val loaders (built once, no resampling needed) ──
    val_loaders = build_val_loaders(args.manifest, args.data_root, args.num_workers)
    val_n = sum(
        len(ld.dataset) for ld in val_loaders.values() if ld is not None
    )
    print(f"Val samples: {val_n}")

    # ── Training loop ──
    global_step = 0
    epoch_iter = range(start_epoch, args.epochs + 1)
    total_epochs = max(0, args.epochs - start_epoch + 1)
    epoch_bar = tqdm(
        epoch_iter,
        total=total_epochs,
        desc="V3 epochs",
        unit="epoch",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for epoch in epoch_bar:
        epoch_started = time.time()

        # Build fresh train loaders each epoch (new random samplers)
        train_loaders = build_train_loaders(
            args.manifest, args.data_root, args.steps_per_group, args.num_workers
        )
        loader_iters = {g: iter(train_loaders[g]) for g in GROUP_ORDER}

        # Per-group accumulators for this epoch
        epoch_loss = {g: [] for g in GROUP_ORDER}
        epoch_dice = {g: [] for g in GROUP_ORDER}

        model.train()
        step_bar = tqdm(
            range(1, args.steps_per_group + 1),
            total=args.steps_per_group,
            desc=f"epoch {epoch}/{args.epochs}",
            unit="step",
            dynamic_ncols=True,
            leave=False,
            disable=args.no_progress,
        )
        for step in step_bar:
            step_postfix = {}
            for group in GROUP_ORDER:
                batch = next(loader_iters[group])
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(x)
                    ce = F.cross_entropy(logits, y)
                    dice_l = dice_loss_from_logits(logits, y)
                    loss = ce + dice_l

                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1

                epoch_loss[group].append(float(loss.item()))
                with torch.no_grad():
                    m = batch_segmentation_metrics(logits, y)
                epoch_dice[group].append(m["dice"])
                step_postfix[f"{group[0]}_loss"] = f"{loss.item():.3f}"
                step_postfix[f"{group[0]}_dice"] = f"{m['dice']:.3f}"

                if global_step % args.log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"epoch={epoch} step={global_step} group={group:>7} "
                        f"loss={loss.item():.4f} dice={m['dice']:.4f} "
                        f"lr={lr:.2e} shape={tuple(x.shape)}"
                    )

            if not args.no_progress:
                step_postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                step_bar.set_postfix(step_postfix)

            if args.smoke and step >= 2:
                break

        # ── Validation ──
        group_metrics, val_overall = validate(model, val_loaders, device)
        val_dice = val_overall["dice"]

        # ── Logging ──
        train_summary = {}
        for g in GROUP_ORDER:
            if epoch_loss[g]:
                train_summary[f"train_{g}_loss"] = float(np.mean(epoch_loss[g]))
                train_summary[f"train_{g}_dice"] = float(np.mean(epoch_dice[g]))
        train_loss = float(np.mean([v for values in epoch_loss.values() for v in values])) if any(epoch_loss.values()) else 0.0

        epoch_summary = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - epoch_started,
            **train_summary,
            "train_loss": train_loss,
            "val_loss": val_overall["loss"],
            "val_dice": val_dice,
            "val_recall": val_overall["recall"],
            "val_precision": val_overall["precision"],
        }
        for g in GROUP_ORDER:
            if g in group_metrics:
                gm = group_metrics[g]
                epoch_summary[f"val_{g}_loss"] = gm["loss"]
                epoch_summary[f"val_{g}_dice"] = gm["dice"]
                epoch_summary[f"val_{g}_recall"] = gm["recall"]
                epoch_summary[f"val_{g}_precision"] = gm["precision"]
                epoch_summary[f"val_{g}_n"] = gm["n"]

        # ── Save checkpoint ──
        checkpoint = args.save_dir / f"epoch_{epoch:03d}.pt"
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_dice": val_dice,
            "metrics": epoch_summary,
            "variant": args.variant,
            "group_patch_shapes": GROUP_PATCH_SHAPES,
            "group_configs": GROUP_CONFIGS,
        }
        if args.keep_epoch_checkpoints:
            torch.save(checkpoint_payload, checkpoint)
        history.append(epoch_summary)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── Print epoch summary ──
        print(f"--- epoch={epoch} train_loss={train_loss:.4f} "
              f"val_dice={val_dice:.4f} val_recall={val_overall['recall']:.4f} "
              f"val_precision={val_overall['precision']:.4f} ---")
        for g in GROUP_ORDER:
            if g in group_metrics:
                gm = group_metrics[g]
                print(f"    {g:>7}: n={gm['n']:4d}  dice={gm['dice']:.4f}  "
                      f"recall={gm['recall']:.4f}  prec={gm['precision']:.4f}")

        # ── Best checkpoint ──
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = args.save_dir / "best.pt"
            torch.save(checkpoint_payload, best_path)
            print(f"  -> best checkpoint (dice={best_val_dice:.4f})")

        scheduler.step()

        latest_path = args.save_dir / "latest.pt"
        torch.save(checkpoint_payload, latest_path)
        if not args.no_progress:
            epoch_bar.set_postfix(
                val_dice=f"{val_dice:.4f}",
                best=f"{best_val_dice:.4f}",
                elapsed=f"{time.time() - epoch_started:.0f}s",
            )

        if args.smoke:
            break

    print(f"Done. best_val_dice={best_val_dice:.4f}")


if __name__ == "__main__":
    main()
