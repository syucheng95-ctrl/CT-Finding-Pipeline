import argparse
import gc
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .input_parser import parse_inputs
from .category_router import route_category
from .hu_expert import run_hu_expert
from .nodule_detector_expert import run_nodule_detector_expert
from .diffuse_expert import propose_gate
from .coordinate_mapper import crop_to_original
from .proposal_fusion import fuse_proposals, compute_compactness, compute_hu_stats


def load_nifti_cache(path: Path, cache: dict) -> nib.Nifti1Image:
    key = str(path)
    if key not in cache:
        cache[key] = nib.load(str(path))
    return cache[key]


def get_zooms(cropped_nii: nib.Nifti1Image) -> tuple:
    return tuple(float(z) for z in cropped_nii.header.get_zooms()[:3])


def process_finding(finding: dict, nifti_cache: dict) -> dict:
    fid = finding["finding_id"]
    category = finding["category"]
    gate_bbox = finding["gate_bbox_hwd"]
    crop_path = finding["cropped_roi_path"]
    source_path = finding["source_image_path"]

    expert = route_category(category)

    if expert == "full_ct_voxtell":
        return {
            "finding_id": fid,
            "prompt": finding["prompt"],
            "category": category,
            "anatomy_target": finding["anatomy_target"],
            "gate_bbox_hwd": gate_bbox,
            "expert": "full_ct_voxtell",
            "fallback": False,
            "fallback_reason": None,
            "n_proposals": 0,
            "proposals": [],
            "use_full_ct_voxtell": True,
            "stage2_from_fullct_components_only": True,
        }

    cropped_nii = load_nifti_cache(crop_path, nifti_cache)
    zooms = get_zooms(cropped_nii)

    # Get real CT shape from source NIfTI header (fast, no data load)
    source_nii = load_nifti_cache(source_path, nifti_cache)
    original_shape = tuple(source_nii.shape)

    bboxes_original = None
    bbox_scores = None
    fallback = False
    fallback_reason = None

    if expert == "hu":
        bboxes_crop = run_hu_expert(cropped_nii, category)
        if bboxes_crop:
            bboxes_original = [crop_to_original(b, gate_bbox) for b in bboxes_crop]

    if expert == "nodule_detector":
        try:
            detections_crop = run_nodule_detector_expert(
                cropped_nii, category, cache_key=str(crop_path),
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"[run_stage0_5] WARN {fid}: nodule detector CUDA OOM; falling back to gate proposal")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            detections_crop = None
            fallback_reason = "nodule_detector_cuda_oom"
        if detections_crop:
            bboxes_original = [crop_to_original(d["bbox"], gate_bbox) for d in detections_crop]
            bbox_scores = [d["score"] for d in detections_crop]

    if bboxes_original is None:
        fallback = True
        bboxes_original = propose_gate(gate_bbox)
        bbox_scores = None
        expert = "diffuse"

    # Only load HU data for experts that need it; diffuse just returns gate
    if expert == "diffuse":
        cropped_hu = None
    else:
        cropped_hu = np.asanyarray(cropped_nii.dataobj).astype(np.float32)

    proposals = fuse_proposals(
        bboxes_original, gate_bbox, original_shape, zooms,
        source_expert=expert, cropped_hu=cropped_hu,
        bbox_scores=bbox_scores,
    )

    for i, p in enumerate(proposals):
        p["proposal_id"] = f"{fid}_p{i:02d}"

    return {
        "finding_id": fid,
        "prompt": finding["prompt"],
        "category": category,
        "anatomy_target": finding["anatomy_target"],
        "gate_bbox_hwd": gate_bbox,
        "expert": expert,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "n_proposals": len(proposals),
        "proposals": proposals,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage0.5 MoE proposal generator")
    parser.add_argument("--predictions", required=True, help="stage0_router_predictions.jsonl")
    parser.add_argument("--crop-groups", required=True, help="stage0_crop_groups.jsonl")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=0, help="limit findings for smoke test")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    crop_path = Path(args.crop_groups)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_file = out_dir / "failed_findings.jsonl"
    if failed_file.exists():
        failed_file.unlink()

    findings = parse_inputs(pred_path, crop_path)
    if args.limit > 0:
        findings = findings[:args.limit]
        print(f"[run_stage0_5] Limited to {args.limit} findings")

    nifti_cache = {}
    results = []
    failed = []

    for i, finding in enumerate(findings):
        try:
            result = process_finding(finding, nifti_cache)
            results.append(result)
        except Exception as exc:
            import traceback
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"[run_stage0_5] ERROR {finding['finding_id']}: {err_msg}")
            failed.append({
                "finding_id": finding["finding_id"],
                "category": finding.get("category", "?"),
                "error": err_msg,
                "traceback": traceback.format_exc(),
            })
            continue
        if (i + 1) % 10 == 0:
            print(f"[run_stage0_5] {i + 1}/{len(findings)} findings done")

    output_file = out_dir / "stage0_5_proposals.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for r in results:
            record = {k: v for k, v in r.items() if k != "prompt"}
            record["prompt"] = r["prompt"]
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"[run_stage0_5] Saved {len(results)} findings → {output_file}")

    if failed:
        with failed_file.open("w", encoding="utf-8") as f:
            for item in failed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[run_stage0_5] {len(failed)} failed findings → {failed_file}")

    # Summary
    n_fallback = sum(1 for r in results if r["fallback"])
    n_hu = sum(1 for r in results if r["expert"] == "hu")
    n_nodule = sum(1 for r in results if r["expert"] == "nodule_detector")
    n_fullct = sum(1 for r in results if r["expert"] == "full_ct_voxtell")
    total_proposals = sum(r["n_proposals"] for r in results)
    print(f"[run_stage0_5] Summary: {len(results)} findings, "
          f"{n_hu} HU expert, {n_nodule} nodule detector, "
          f"{n_fullct} full_ct_voxtell, {n_fallback} fallback, "
          f"{total_proposals} total proposals "
          f"({total_proposals / max(1, len(results)):.1f} avg)")


if __name__ == "__main__":
    main()
