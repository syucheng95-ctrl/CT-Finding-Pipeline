"""⑤ Pipeline — end-to-end: ROI → standardised report.

Usage:
    python -m stage2_5.pipeline  # run on all Stage2 ROIs
"""

import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

_PROJ = Path(__file__).resolve().parent.parent


def _load_context_map():
    """Load structured findings → {finding_id: clinical_context}."""
    ctx_path = _PROJ / "outputs" / "structured_findings.jsonl"
    if not ctx_path.exists():
        return {}
    ctx_map = {}
    with open(ctx_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ctx_map[r["finding_id"]] = r
    return ctx_map


def process_single_roi(roi_row, context_map=None):
    """Process a single ROI through the full pipeline.

    Args:
        roi_row: dict from roi_manifest.jsonl (one JSONL line).

    Returns:
        dict: standardized report.
    """
    from stage2_5.metrics import compute_all_metrics
    from stage2_5.report import generate_report

    roi_root = Path(roi_row["_roi_root"]) if "_roi_root" in roi_row else (
        Path(__file__).resolve().parent.parent / "outputs" / "stage2"
    )

    # ── Load data ──
    img_path = roi_root / (roi_row.get("roi_image_path") or roi_row.get("roi_image", ""))
    mask_path = roi_root / (roi_row.get("roi_mask_path") or roi_row.get("roi_mask", ""))
    img = nib.load(str(img_path))
    mask = nib.load(str(mask_path))
    ct = np.asanyarray(img.dataobj).astype(np.float32).transpose(2, 0, 1)  # HWD → DHW
    m = np.asanyarray(mask.dataobj).astype(np.uint8).transpose(2, 0, 1)
    spacing = img.header.get_zooms()

    # ── ① Classify ──
    # Use ground-truth category from manifest (already labeled by upstream pipeline).
    # For production inference, load the classifier model and call _classify_single().
    category = roi_row.get("category", "2c")

    # ── ② Metrics ──
    metrics = compute_all_metrics(ct, m, spacing, category)

    # ── ③ Clinical context ──
    ctx = {}
    if context_map:
        fid = roi_row.get("parent_sample_id", "")
        ctx = context_map.get(fid, {})

    # ── ④ Report ──
    report = generate_report(metrics, ctx, finding_id=roi_row.get("parent_sample_id", ""),
                             category=category)
    report["roi_id"] = roi_row.get("component_sample_id", "")

    return report


def process_batch(roi_rows, context_map=None):
    """Process multiple ROIs, yielding reports one at a time."""
    context_map = context_map or _load_context_map()
    for row in roi_rows:
        yield process_single_roi(row, context_map)


def main():
    """Stage2.5 CLI: process Stage2 ROIs and output standardized reports.

    Usage:
        python -m stage2_5.pipeline                          # smoke test (3 ROIs)
        python -m stage2_5.pipeline --all                    # all ROIs
        python -m stage2_5.pipeline --all --output reports.jsonl  # save to file
        python -m stage2_5.pipeline --finding finding_004530 # single finding
    """
    import argparse
    parser = argparse.ArgumentParser(description="Stage2.5 ROI quant pipeline")
    parser.add_argument("--all", action="store_true", help="Process all ROIs")
    parser.add_argument("--finding", default="", help="Process a specific finding ID")
    parser.add_argument("--limit", type=int, default=0, help="Limit N ROIs")
    parser.add_argument("--output", default="", help="Output JSONL path (default: stdout)")
    args = parser.parse_args()

    from stage2_5.classifier.config import ROI_MANIFEST

    with open(ROI_MANIFEST, encoding="utf-8") as f:
        all_rows = [json.loads(l) for l in f if l.strip()]

    if not all_rows:
        print("No ROIs found in", ROI_MANIFEST)
        return

    # Select ROIs
    if args.finding:
        rows = [r for r in all_rows if r.get("parent_sample_id") == args.finding]
        if not rows:
            print(f"No ROIs found for finding: {args.finding}")
            return
    elif args.all:
        rows = all_rows
    else:
        # Smoke test: one per metric group
        groups = {"1a": "airway_change", "1c": "lung_opacity", "2c": "lung_opacity",
                  "2e": "pleural", "2d": "lung_nodule"}
        rows = []
        seen = set()
        for r in all_rows:
            cat = r["category"]
            if cat in groups and cat not in seen:
                rows.append(r)
                seen.add(cat)
        print(f"Smoke test: {len(rows)} ROIs (one per metric group)")

    if args.limit:
        rows = rows[: args.limit]

    print(f"Processing {len(rows)} ROIs...")

    context_map = _load_context_map()
    t0 = time.time()
    reports = []

    out_f = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.output, "w", encoding="utf-8")

    for i, row in enumerate(rows):
        try:
            report = process_single_roi(row, context_map=context_map)
            reports.append(report)
            if out_f:
                out_f.write(json.dumps(report, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(rows)}] ROIs done ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [ERROR] {row.get('component_sample_id', '?')}: {e}")

    if out_f:
        out_f.close()

    elapsed = time.time() - t0
    print(f"\nProcessed {len(reports)}/{len(rows)} ROIs in {elapsed:.0f}s")
    if args.output:
        print(f"Reports saved to {args.output}")


if __name__ == "__main__":
    main()
