"""Build STU-Net-compatible manifest from the full ROI dataset.

Split scheme (方案 A):
  - Original ReXGroundingCT val cases (50, board-certified radiologist labels)
    → holdout (untouched during training, for final benchmark evaluation)
  - Remaining ~2,992 cases → case-level stratified 80/20 split (seed=42)
    → train / val

Output: a single JSONL manifest with split in {train, val, holdout}.
ReXStage2ROIPatchDataset filters by split, so holdout rows are ignored during training.

Usage:
  python scripts/build_stage2_full_manifest.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


PATCH_SIZE_XYZ = [96, 96, 48]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_case_stratified_split(rows: list[dict], seed: int = 42) -> dict[str, str]:
    """Case-level stratified 80/20 split (copied from Stage 2.5)."""
    case_rows: dict[str, list[dict]] = {}
    for r in rows:
        case_rows.setdefault(r["case_name"], []).append(r)

    cases = sorted(case_rows)
    case_primary_cats = []
    for c in cases:
        cat_counts = Counter(r["category"] for r in case_rows[c])
        case_primary_cats.append(cat_counts.most_common(1)[0][0])

    train_cases, val_cases = train_test_split(
        cases, test_size=0.20, stratify=case_primary_cats, random_state=seed,
    )
    split_map = {}
    for c in train_cases:
        split_map[c] = "train"
    for c in val_cases:
        split_map[c] = "val"
    return split_map


def adapt_row(row: dict, split_val: str) -> dict:
    return {
        "component_sample_id": row["component_id"],
        "parent_sample_id": row.get("parent_id", ""),
        "split": split_val,
        "prompt": row.get("prompt", ""),
        "roi_image_path": row["roi_image"],
        "roi_mask_path": row["roi_mask"],
        "source_image": row.get("source_image", ""),
        "source_label": row.get("parent_label", ""),
        "array_axis_order": "xyz",
        "source_image_shape_xyz": None,
        "patch_size_xyz": PATCH_SIZE_XYZ,
        "voxel_count": row["component_voxels"],
        "category": row.get("category", ""),
        "case_name": row["case_name"],
        "roi_shape_hwd": row.get("roi_shape_hwd", row.get("roi_shape_xyz", [0, 0, 0])),
    }


def main():
    p = argparse.ArgumentParser(description="Build STU-Net manifest with holdout (方案 A)")
    p.add_argument("--in-manifest", type=Path,
                   default=Path("full_dataset_work/stage2_roi_50case/roi_manifest.jsonl"))
    p.add_argument("--out-manifest", type=Path,
                   default=Path("full_dataset_work/stage2_roi_50case/stage2_train_manifest.jsonl"))
    args = p.parse_args()

    print(f"Reading: {args.in_manifest}")
    all_rows = load_jsonl(args.in_manifest)
    print(f"  {len(all_rows)} ROI samples total")

    # Step 1: Extract original val cases → holdout
    holdout_cases = set(r["case_name"] for r in all_rows if r["split"] == "val")
    holdout_rows = [r for r in all_rows if r["case_name"] in holdout_cases]
    rest_rows = [r for r in all_rows if r["case_name"] not in holdout_cases]
    print(f"  Holdout: {len(holdout_rows)} samples, {len(holdout_cases)} cases")
    print(f"  Remaining for train/val: {len(rest_rows)} samples, {len(set(r['case_name'] for r in rest_rows))} cases")

    # Step 2: Case-stratified 80/20 on remaining
    print("\nBuilding case-level stratified split (80/20, seed=42)...")
    split_map = build_case_stratified_split(rest_rows)
    n_train = sum(1 for v in split_map.values() if v == "train")
    n_val = sum(1 for v in split_map.values() if v == "val")
    print(f"  Train cases: {n_train}, Val cases: {n_val}")

    # Step 3: Adapt all rows
    out_rows = []
    for r in holdout_rows:
        out_rows.append(adapt_row(r, "holdout"))
    for r in rest_rows:
        out_rows.append(adapt_row(r, split_map[r["case_name"]]))

    # Summary
    splits = Counter(r["split"] for r in out_rows)
    print(f"\nFinal split:")
    print(f"  Train:   {splits['train']} samples")
    print(f"  Val:     {splits['val']} samples")
    print(f"  Holdout: {splits['holdout']} samples")

    cat_counts = Counter((r["category"], r["split"]) for r in out_rows)
    print(f"\nPer-category:")
    for cat in sorted(set(c for c, _ in cat_counts)):
        tr = cat_counts[(cat, "train")]
        va = cat_counts[(cat, "val")]
        ho = cat_counts[(cat, "holdout")]
        print(f"  {cat}: train={tr:5d}  val={va:5d}  holdout={ho:4d}")

    print(f"\nWriting: {args.out_manifest}")
    write_jsonl(args.out_manifest, out_rows)
    print("Done.")


if __name__ == "__main__":
    main()
