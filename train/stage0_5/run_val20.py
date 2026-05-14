"""Run Stage0.5 on all val_20 cases. Combines cropping + proposal generation."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

STAGE0_V2 = (Path(__file__).resolve().parent.parent.parent
              / "stage0_workspace" / "stage0_v2" / "router")
if str(STAGE0_V2) not in sys.path:
    sys.path.insert(0, str(STAGE0_V2))

from router_utils import read_jsonl, write_jsonl
from anatomy_expert import (
    TotalSegLobeCache, parse_anatomy, anatomy_to_bbox, anatomy_record,
    LOBE_LABELS, SIDE_TARGETS,
)
from evaluate_stage0_policy_recall import (
    LungCache, bbox_volume, bbox_for_anatomy_groups,
    policy_to_bbox, image_path_for, resolve_path, union_bboxes,
)

from proposal_generator.category_router import route_category
from proposal_generator.hu_expert import run_hu_expert
from proposal_generator.nodule_detector_expert import run_nodule_detector_expert
from proposal_generator.diffuse_expert import propose_gate
from proposal_generator.coordinate_mapper import crop_to_original
from proposal_generator.proposal_fusion import fuse_proposals

MARGINS = {
    "conservative": [60, 60, 50],
    "moderate": [40, 40, 30],
    "aggressive": [20, 20, 20],
}


def parse_margin(text):
    return [float(x) for x in text.split(",")]


def crop_nifti(nii, bbox):
    h0, h1, w0, w1, d0, d1 = bbox
    arr = np.asanyarray(nii.dataobj)
    cropped = arr[h0:h1, w0:w1, d0:d1].copy()
    affine = nii.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(nii.affine, [h0, w0, d0])
    return nib.Nifti1Image(cropped, affine, nii.header)


_IMAGE_ROOTS: list[Path] = []


def _find_ct(case_name: str) -> Path:
    """Find CT image for a case. Checks --image-root paths first, then fallback dirs."""
    candidates = [d / case_name for d in _IMAGE_ROOTS]
    # Fallback: check stage1_verification data dirs relative to project root
    project_root = STAGE0_V2.parent.parent.parent  # 生医工大赛/
    candidates += [
        project_root / "stage1_verification" / "data" / "images" / case_name,
        project_root / "stage1_verification" / "data" / "images_flat" / case_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"CT not found for {case_name}. Searched: {candidates}")


def process_case(case_name, case_preds, manifest, data_root,
                 lung_cache, totalseg_cache, margin_mm):
    """Run Stage0.5 for one case. Returns per-finding result dicts."""
    image_path = _find_ct(case_name)
    nii = nib.load(str(image_path))
    shape = nii.shape
    zooms = nii.header.get_zooms()[:3]

    lung_info = lung_cache.get(image_path, shape, nii.affine, "fixed")
    seg = totalseg_cache.get_seg(image_path, shape)

    # Compute anatomy gates for each finding (same logic as Stage0 v2)
    per_prompt_bboxes = []
    for p in case_preds:
        p["final_policy"] = p.get("final_policy", "both_conservative")
        p["final_tightness"] = p.get("final_tightness", "conservative")
        p.setdefault("laterality", "unknown")

        fallback_bbox = policy_to_bbox(
            p["final_policy"], lung_info, shape, zooms, margin_mm,
        )
        p.update(anatomy_record(p["prompt"], p["laterality"], p["final_tightness"]))
        decision = parse_anatomy(p["prompt"], p["laterality"])
        bbox = anatomy_to_bbox(
            decision, seg, shape, zooms, margin_mm[p["final_tightness"]],
        )
        if bbox is None:
            bbox = fallback_bbox
        per_prompt_bboxes.append(bbox)

    finding_bboxes, groups = bbox_for_anatomy_groups(
        case_preds, per_prompt_bboxes, "policy_group",
    )

    # Group findings by crop for caching
    by_crop_key = defaultdict(list)
    for p, bbox in zip(case_preds, per_prompt_bboxes):
        crop_key = tuple(bbox) if bbox else None
        by_crop_key[crop_key].append((p, bbox))

    # Process each finding through Stage0.5
    results = []
    nii_cache = {}
    for p, gate_bbox in zip(case_preds, per_prompt_bboxes):
        fid = p["id"]
        category = p.get("pred_category", p.get("category", "1f"))
        expert = route_category(category)
        if gate_bbox is None:
            gate_bbox = [0, shape[0], 0, shape[1], 0, shape[2]]

        try:
            # Crop CT on-the-fly
            cropped_nii = crop_nifti(nii, gate_bbox)
            cropped_shape = cropped_nii.shape

            bboxes_original = None
            bbox_scores = None
            fallback = False
            used_expert = expert

            if expert == "hu":
                bboxes_crop = run_hu_expert(cropped_nii, category)
                if bboxes_crop:
                    bboxes_original = [crop_to_original(b, gate_bbox) for b in bboxes_crop]

            if expert == "nodule_detector":
                crop_key = str(tuple(gate_bbox))
                dets = run_nodule_detector_expert(
                    cropped_nii, category, cache_key=f"{case_name}:{crop_key}",
                )
                if dets:
                    bboxes_original = [crop_to_original(d["bbox"], gate_bbox) for d in dets]
                    bbox_scores = [d["score"] for d in dets]

            if bboxes_original is None:
                fallback = True
                bboxes_original = propose_gate(gate_bbox)
                bbox_scores = None
                used_expert = "diffuse"

            cropped_hu = None if used_expert == "diffuse" else np.asanyarray(
                cropped_nii.dataobj,
            ).astype(np.float32)
            proposals = fuse_proposals(
                bboxes_original, gate_bbox, tuple(shape), zooms,
                source_expert=used_expert, cropped_hu=cropped_hu,
                bbox_scores=bbox_scores,
            )
            for i, prop in enumerate(proposals):
                prop["proposal_id"] = f"{fid}_p{i:02d}"

            results.append({
                "finding_id": fid,
                "prompt": p["prompt"],
                "category": category,
                "anatomy_target": p.get("anatomy_target", "unknown"),
                "gate_bbox_hwd": gate_bbox,
                "expert": used_expert,
                "fallback": fallback,
                "n_proposals": len(proposals),
                "proposals": proposals,
            })
        except Exception as exc:
            import traceback
            print(f"  ERROR {fid}: {exc}")
            traceback.print_exc()
            continue

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-root", nargs="*", default=[],
                        help="Additional directories to search for CT images")
    parser.add_argument("--totalseg-device", default="gpu")
    parser.add_argument("--totalseg-fast", action="store_true", default=True)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--conservative-margin", default="60,60,50")
    parser.add_argument("--moderate-margin", default="40,40,30")
    parser.add_argument("--aggressive-margin", default="20,20,20")
    args = parser.parse_args()

    # Populate image search roots from CLI
    for d in args.image_root:
        _IMAGE_ROOTS.append(Path(d))

    pred_path = Path(args.predictions)
    manifest_path = resolve_path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    margin_mm = {
        "conservative": parse_margin(args.conservative_margin),
        "moderate": parse_margin(args.moderate_margin),
        "aggressive": parse_margin(args.aggressive_margin),
    }

    predictions = read_jsonl(pred_path)
    manifest = {r["id"]: r for r in read_jsonl(manifest_path)}
    predictions = [p for p in predictions if p["id"] in manifest]
    data_root = manifest_path.parent

    by_case = defaultdict(list)
    for p in predictions:
        by_case[manifest[p["id"]]["case_name"]].append(p)

    cases = sorted(by_case.items())
    if args.limit_cases > 0:
        cases = cases[:args.limit_cases]
        print(f"Limited to {args.limit_cases} cases")

    print(f"Running Stage0.5 on {len(cases)} cases ({sum(len(v) for _, v in cases)} findings)")

    lung_cache = LungCache()
    totalseg_cache = TotalSegLobeCache(
        fast=args.totalseg_fast, device=args.totalseg_device,
    )

    all_results = []
    t0 = time.time()
    for idx, (case_name, case_preds) in enumerate(cases):
        t_case = time.time()
        print(f"[{idx+1}/{len(cases)}] {case_name} ({len(case_preds)} findings)...", end=" ", flush=True)
        try:
            results = process_case(
                case_name, case_preds, manifest, data_root,
                lung_cache, totalseg_cache, margin_mm,
            )
            all_results.extend(results)
            n_hu = sum(1 for r in results if r["expert"] == "hu")
            n_nodule = sum(1 for r in results if r["expert"] == "nodule_detector")
            n_fallback = sum(1 for r in results if r["fallback"])
            n_proposals = sum(r["n_proposals"] for r in results)
            elapsed = time.time() - t_case
            print(f"{len(results)} findings, {n_hu}H {n_nodule}N {n_fallback}F, "
                  f"{n_proposals} proposals, {elapsed:.0f}s")
            # Clear per-case cache to avoid unbounded memory growth
            image_path = _find_ct(case_name)
            lung_cache.cache.pop(str(image_path), None)
            totalseg_cache.cache.pop(str(image_path), None)
        except Exception as exc:
            import traceback
            print(f"FAILED: {exc}")
            traceback.print_exc()

    total_elapsed = time.time() - t0
    output_file = out_dir / "stage0_5_proposals.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for r in all_results:
            record = {k: v for k, v in r.items() if k != "prompt"}
            record["prompt"] = r["prompt"]
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    n_hu = sum(1 for r in all_results if r["expert"] == "hu")
    n_nodule = sum(1 for r in all_results if r["expert"] == "nodule_detector")
    n_fallback = sum(1 for r in all_results if r["fallback"])
    print(f"\nDone in {total_elapsed:.0f}s: {len(all_results)} findings, "
          f"{n_hu} HU, {n_nodule} nodule, {n_fallback} fallback → {output_file}")


if __name__ == "__main__":
    main()
