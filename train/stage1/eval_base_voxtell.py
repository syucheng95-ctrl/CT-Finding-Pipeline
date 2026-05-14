"""Sliding-window evaluation for the original VoxTell checkpoint without DoRA."""

import argparse
import json
import time
from pathlib import Path

import torch

from config import build_voxtell_from_checkpoint
from eval import (
    DoRAEvalPredictor,
    NibabelIOWithReorient,
    add_metric,
    empty_bucket,
    load_gt_mask,
    load_jsonl,
    metrics,
    parse_thresholds,
    sigmoid_np,
    summarize_bucket,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs_fixed/exp_base/eval.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--text-model", type=str, default=None)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Use precomputed training embeddings/*.pt instead of online Qwen embeddings.",
    )
    parser.add_argument("--thresholds", default="0.5")
    parser.add_argument("--per-finding-output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    thresholds = parse_thresholds(args.thresholds)

    model = build_voxtell_from_checkpoint(args.model_dir, device=torch.device("cpu"))
    model = model.to(device).eval()

    local_text_model = (
        Path(__file__).resolve().parent.parent
        / "生医工大赛-demo"
        / "data"
        / "modelscope_models"
        / "Qwen"
        / "Qwen3-Embedding-4B"
    )
    text_model_name = args.text_model or (
        str(local_text_model) if local_text_model.exists() else "Qwen/Qwen3-Embedding-4B"
    )
    predictor = DoRAEvalPredictor(
        model_dir=str(args.model_dir),
        device=device,
        text_encoding_model=text_model_name,
        embedding_dir=args.embedding_dir,
    )
    predictor.network = model

    reader = NibabelIOWithReorient()
    rows = load_jsonl(args.manifest)
    case_map = {}
    for row in rows:
        case_map.setdefault(row["case_name"], []).append(row)
    cases = sorted(case_map.items())
    if args.limit_cases:
        cases = cases[: args.limit_cases]

    buckets = {str(t): empty_bucket() for t in thresholds}
    per_finding_rows = []
    skipped_images = missing_masks = failed_cases = 0

    for ci, (case_name, case_rows) in enumerate(cases, 1):
        image_path = args.image_root / case_name
        if not image_path.exists():
            print(f"  [SKIP] {case_name}")
            skipped_images += 1
            continue

        t0 = time.time()
        image, _ = reader.read_images([str(image_path)])
        prompts = [row["prompt"] for row in case_rows]
        predictor.current_finding_ids = [row["id"] for row in case_rows]
        try:
            logits = predictor.predict_single_image_logits(image, prompts)
        except Exception as exc:
            print(f"  [ERR] {case_name}: {exc}")
            failed_cases += 1
            continue

        for i, row in enumerate(case_rows):
            mask_rel = row["label"].replace("labels_finding/", "").replace("\\", "/")
            mask_path = args.label_root / mask_rel
            if not mask_path.exists():
                print(f"  [SKIP] missing mask: {mask_path}")
                missing_masks += 1
                continue

            gt = load_gt_mask(mask_path, reader)
            prob = sigmoid_np(logits[i])
            finding_diag = {
                "case_name": case_name,
                "id": row["id"],
                "category": row.get("category", ""),
                "prompt": row["prompt"],
                "gt_voxels": int(gt.sum()),
                "max_prob": float(prob.max()),
                "mean_prob": float(prob.mean()),
                "gt_max_prob": float(prob[gt > 0].max()) if gt.any() else 0.0,
                "gt_mean_prob": float(prob[gt > 0].mean()) if gt.any() else 0.0,
                "thresholds": {},
            }
            for threshold in thresholds:
                key = str(threshold)
                d, r, p, pv, gv = metrics(prob > threshold, gt)
                add_metric(buckets[key], d, r, p, pv, gv)
                finding_diag["thresholds"][key] = {
                    "dice": d,
                    "recall": r,
                    "precision": p,
                    "pred_voxels": pv,
                }
            per_finding_rows.append(finding_diag)

        print(f"[{ci}/{len(cases)}] {case_name} ({len(prompts)} prompts, {time.time() - t0:.0f}s)")

    n_evaluated = sum(bucket["n"] for bucket in buckets.values()) // max(len(buckets), 1)
    if n_evaluated == 0:
        raise RuntimeError(
            "No findings were evaluated. "
            f"skipped_images={skipped_images}, missing_masks={missing_masks}, failed_cases={failed_cases}"
        )

    threshold_results = {key: summarize_bucket(bucket) for key, bucket in buckets.items()}
    default_key = str(thresholds[0])
    default_res = threshold_results[default_key]
    result = {
        "exp": "base",
        "checkpoint": str(args.model_dir / "fold_0" / "checkpoint_final.pth"),
        "n_findings": default_res["n_findings"],
        "dice": default_res["dice"],
        "recall": default_res["recall"],
        "precision": default_res["precision"],
        "pred_voxels": default_res["pred_voxels"],
        "gt_voxels": default_res["gt_voxels"],
        "vol_ratio": default_res["vol_ratio"],
        "thresholds": threshold_results,
        "skipped_images": skipped_images,
        "missing_masks": missing_masks,
        "failed_cases": failed_cases,
    }

    print("\nExp base:")
    for key, item in threshold_results.items():
        print(
            f"  th={key}: Dice={item['dice']:.4f} Recall={item['recall']:.4f} "
            f"Precision={item['precision']:.4f} VolRatio={item['vol_ratio']:.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {args.output}")

    per_finding_out = args.per_finding_output or args.output.with_suffix(".per_finding.jsonl")
    with per_finding_out.open("w", encoding="utf-8") as f:
        for row in per_finding_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved per-finding diagnostics to {per_finding_out}")


if __name__ == "__main__":
    main()
