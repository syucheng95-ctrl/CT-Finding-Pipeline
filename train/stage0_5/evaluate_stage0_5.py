import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

STAGE0_V2 = (Path(__file__).resolve().parent.parent.parent
              / "stage0_workspace" / "stage0_v2" / "router")
if str(STAGE0_V2) not in sys.path:
    sys.path.insert(0, str(STAGE0_V2))

from router_utils import read_jsonl, write_jsonl, write_csv, write_json
from evaluate_stage0_policy_recall import (
    bbox_volume,
    union_bboxes,
    label_path_for,
    resolve_path,
)

from .config import CATEGORY_LABELS, CATEGORY_NAMES, CATEGORY_TO_EXPERT


def load_proposals(path: Path) -> list:
    return read_jsonl(path)


def proposals_union_mask(proposal_bboxes: list, shape: tuple) -> np.ndarray:
    """Binary mask where any proposal voxel is True. Avoids giant bounding-box inflation."""
    mask = np.zeros(shape, dtype=bool)
    for bbox in proposal_bboxes:
        h0, h1, w0, w1, d0, d1 = bbox
        H, W, D = shape
        h0, h1 = max(0, h0), min(H, h1)
        w0, w1 = max(0, w0), min(W, w1)
        d0, d1 = max(0, d0), min(D, d1)
        if h0 < h1 and w0 < w1 and d0 < d1:
            mask[h0:h1, w0:w1, d0:d1] = True
    return mask


def compute_recall(proposal_bboxes: list, label_nii: nib.Nifti1Image) -> dict:
    label = np.asanyarray(label_nii.dataobj) > 0
    total = int(label.sum())

    if not proposal_bboxes:
        inside = 0
    else:
        union_mask = proposals_union_mask(proposal_bboxes, label.shape)
        inside = int(label[union_mask].sum())

    recall = 1.0 if total == 0 else inside / total
    return {"gt_voxels": total, "inside_voxels": inside, "recall": recall}


def compute_volume_ratio(proposal_bboxes: list, ct_shape: tuple) -> float:
    if not proposal_bboxes:
        return 0.0
    union_mask = proposals_union_mask(proposal_bboxes, ct_shape)
    ct_vol = max(1, int(np.prod(ct_shape)))
    return int(union_mask.sum()) / ct_vol


def summarize(values: list) -> dict:
    if not values:
        return {"n": 0, "mean": None}
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p90_count": int(np.sum(arr >= 0.90)),
        "strict_count": int(np.sum(arr >= 0.999)),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage0.5 evaluation")
    parser.add_argument("--proposals", required=True, help="stage0_5_proposals.jsonl")
    parser.add_argument("--manifest", required=True, help="manifest JSONL with id/case_name/label")
    parser.add_argument("--out-dir", required=True, help="output directory")
    args = parser.parse_args()

    proposals_path = Path(args.proposals)
    manifest_path = resolve_path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proposals = load_proposals(proposals_path)
    manifest = {r["id"]: r for r in read_jsonl(manifest_path)}
    data_root = manifest_path.parent

    proposals = [p for p in proposals if p["finding_id"] in manifest]

    by_category = defaultdict(list)
    finding_rows = []

    for p in proposals:
        fid = p["finding_id"]
        record = manifest[fid]
        label_path = label_path_for(record, data_root)

        try:
            label_nii = nib.load(str(label_path))
        except Exception:
            print(f"[eval] Cannot load label for {fid}: {label_path} — skipping")
            continue

        ct_shape = tuple(label_nii.shape)
        proposal_bboxes = [pr["proposal_bbox_hwd"] for pr in p["proposals"]]
        recall_info = compute_recall(proposal_bboxes, label_nii)
        vol_ratio = compute_volume_ratio(proposal_bboxes, ct_shape)

        category = p["category"]
        hit = 1.0 if recall_info["recall"] >= 0.05 else 0.0

        finding_rows.append({
            "finding_id": fid,
            "case_name": record.get("case_name", ""),
            "category": category,
            "expert": p["expert"],
            "fallback": p["fallback"],
            "n_proposals": p["n_proposals"],
            "gt_voxels": recall_info["gt_voxels"],
            "inside_voxels": recall_info["inside_voxels"],
            "gt_recall": recall_info["recall"],
            "vol_ratio": vol_ratio,
            "hit": hit,
        })
        by_category[category].append(recall_info["recall"])

    # Per-category summary
    cat_rows = []
    for cat in CATEGORY_LABELS:
        if cat not in by_category:
            continue
        rec = summarize(by_category[cat])
        cat_rows.append({
            "category": cat,
            "name": CATEGORY_NAMES[cat],
            "expert": CATEGORY_TO_EXPERT.get(cat, "diffuse"),
            "n_findings": rec["n"],
            "mean_recall": rec["mean"],
            "p05_recall": rec["p05"],
            "hit_rate_p90": rec["p90_count"] / max(1, rec["n"]),
        })

    # Overall
    all_recalls = [r["gt_recall"] for r in finding_rows]
    all_vol_ratios = [r["vol_ratio"] for r in finding_rows]
    all_n_proposals = [r["n_proposals"] for r in finding_rows]
    overall = {
        "n_findings": len(finding_rows),
        "mean_recall": float(np.mean(all_recalls)) if all_recalls else None,
        "mean_vol_ratio": float(np.mean(all_vol_ratios)) if all_vol_ratios else None,
        "mean_n_proposals": float(np.mean(all_n_proposals)) if all_n_proposals else None,
        "finding_hit_rate": float(np.mean([r["hit"] for r in finding_rows])) if finding_rows else None,
        "n_fallback": sum(1 for r in finding_rows if r["fallback"]),
    }

    write_jsonl(finding_rows, out_dir / "finding_recall.jsonl")
    write_csv(cat_rows, out_dir / "category_summary.csv")
    write_csv(finding_rows, out_dir / "finding_recall.csv")
    write_json(overall, out_dir / "summary.json")

    print(f"[eval] {overall['n_findings']} findings evaluated")
    print(f"[eval] Mean recall: {overall['mean_recall']:.4f}")
    print(f"[eval] Mean vol ratio: {overall['mean_vol_ratio']:.4f}")
    print(f"[eval] Finding hit rate: {overall['finding_hit_rate']:.4f}")
    print(f"[eval] Mean proposals/finding: {overall['mean_n_proposals']:.1f}")
    print(f"[eval] Fallback count: {overall['n_fallback']}")

    print("\nPer-category:")
    for r in cat_rows:
        print(f"  {r['category']} {r['name']:30s} n={r['n_findings']:2d}  "
              f"recall={r['mean_recall']:.3f}  p05={r['p05_recall']:.3f}  "
              f"expert={r['expert']}")


if __name__ == "__main__":
    main()
