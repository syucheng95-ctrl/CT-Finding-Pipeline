"""Stage1 runner: Per-proposal VoxTell text-consistency verifier.

Reads Stage0's cropped ROIs + Stage0.5's proposals, runs VoxTell on each
cropped ROI with the finding text, and scores each proposal by activation
inside its bounding box. Outputs verified_proposals.jsonl + coarse masks.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# VoxTell adapter (handles workspace path setup + NibabelIO import)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "external"))
from voxtell_inference import get_nibabel_io_with_reorient
NibabelIOWithReorient = get_nibabel_io_with_reorient()

from utils import load_config, load_jsonl, resolve, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 VoxTell verifier runner")
    parser.add_argument("--config", default="config.yaml", help="Path to pipeline config")
    parser.add_argument("--limit-findings", type=int, default=0, help="Limit N findings (0=all)")
    parser.add_argument("--finding-ids", default="", help="Comma-separated finding IDs to run")
    parser.add_argument("--append-existing", action="store_true",
                        help="Merge results into an existing verified_proposals.jsonl")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    upload_dir = Path(config["_upload_dir"])
    out_dir = Path(resolve(config, "outputs.stage1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    stage0_out = Path(resolve(config, "outputs.stage0"))
    stage0_5_out = Path(resolve(config, "outputs.stage0_5"))
    crop_groups_file = stage0_out / "stage0_crop_groups.jsonl"
    proposals_file = stage0_5_out / "stage0_5_proposals.jsonl"

    if not proposals_file.exists():
        print(f"[ERROR] Proposals not found: {proposals_file}")
        print("  Run Stage0.5 first.")
        raise SystemExit(1)

    # Load data
    crop_groups = load_jsonl(crop_groups_file)
    proposals = load_jsonl(proposals_file)

    # Build index: finding_id → crop info
    finding_to_crop: dict[str, dict] = {}
    for g in crop_groups:
        for fid in g.get("finding_ids", []):
            finding_to_crop[fid] = {
                "cropped_roi_path": g["image"],
                "gate_bbox_hwd": g["bbox_hwd"],
                "source_image": g.get("source_image", ""),
            }

    # Build finding index
    prop_by_finding: dict[str, dict] = {}
    for p in proposals:
        prop_by_finding[p["finding_id"]] = p

    # Filter to findings that have both crops and proposals
    findings = [(fid, prop_by_finding[fid])
                for fid in prop_by_finding
                if fid in finding_to_crop]
    print(f"Matched {len(findings)} findings with crops + proposals "
          f"({len(prop_by_finding) - len(findings)} missing crops)")

    if args.finding_ids:
        wanted = {fid.strip() for fid in args.finding_ids.split(",") if fid.strip()}
        findings = [(fid, row) for fid, row in findings if fid in wanted]
        missing = sorted(wanted - {fid for fid, _ in findings})
        if missing:
            print(f"[WARN] Requested finding IDs not found in crops + proposals: {missing}")
        print(f"Filtered to {len(findings)} requested findings")

    if args.limit_findings:
        findings = findings[:args.limit_findings]

    # ── Split: full_ct_voxtell vs normal ──
    fullct_findings = [(fid, row) for fid, row in findings
                       if row.get("expert") == "full_ct_voxtell"]
    normal_findings = [(fid, row) for fid, row in findings
                       if row.get("expert") != "full_ct_voxtell"]
    print(f"FullCT VoxTell: {len(fullct_findings)} findings, "
          f"Pipeline: {len(normal_findings)} findings")

    # Load verifier (once)
    from _verifier import Verifier
    verifier = Verifier(
        model_dir=resolve(config, "models.voxtell"),
        text_model=resolve(config, "models.qwen_embedding"),
        device=args.device,
        project_root=upload_dir,
    )

    verified_rows = []
    total_t0 = time.time()

    # ── Process full_ct_voxtell findings: FullCT VoxTell, no crop/proposal ──
    ct_images_dir = Path(resolve(config, "data.ct_images"))
    for fid, prop_row in fullct_findings:
        crop_info = finding_to_crop.get(fid, {})
        source_ct = crop_info.get("source_image", "")
        ct_path = Path(source_ct)
        if not ct_path.exists():
            ct_path = ct_images_dir / (prop_row.get("case_name", "") + ".nii.gz")
        if not ct_path.exists():
            ct_path = ct_images_dir / (prop_row.get("case_name", ""))
        if not ct_path.exists():
            print(f"  [SKIP] {fid}: full CT not found")
            continue

        # Load with NibabelIOWithReorient to match VoxTell training orientation
        reader = NibabelIOWithReorient()
        raw_nii = nib.load(str(ct_path))
        raw_shape = raw_nii.shape[:3]  # (H, W, D)
        reoriented, _ = reader.read_images([str(ct_path)])

        # Match FullCT baseline: pass reoriented directly (same as eval.py)
        logits = verifier.predictor.predict_single_image_logits(
            reoriented.copy(), [prop_row["prompt"]],
        )
        clipped = np.clip(logits[0], -50, 50)
        prob = 1.0 / (1.0 + np.exp(-clipped))
        binary = (prob > 0.3).astype(np.uint8)

        mask_out_dir = out_dir / "masks" / fid
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_out_dir / "full_ct_coarse.nii.gz"
        nib.save(nib.Nifti1Image(binary, np.eye(4)), str(mask_path))
        prob_path = mask_out_dir / "full_ct_prob.nii.gz"
        nib.save(nib.Nifti1Image(prob.astype(np.float32), np.eye(4)), str(prob_path))

        full_bbox = [0, int(raw_shape[0]), 0, int(raw_shape[1]), 0, int(raw_shape[2])]
        verified_rows.append({
            "finding_id": fid,
            "prompt": prop_row["prompt"],
            "category": prop_row["category"],
            "expert": "full_ct_voxtell",
            "use_full_ct_voxtell": True,
            "stage2_from_fullct_components_only": True,
            "verified": [{
                "proposal_id": f"{fid}_fullct",
                "source_expert": "full_ct_voxtell",
                "proposal_bbox_hwd": full_bbox,
                "coarse_mask_path": str(mask_path),
                "coarse_prob_path": str(prob_path),
            }],
            "rejected": [],
            "fallback": False,
            "gate_bbox_hwd": full_bbox,
        })
        torch.cuda.empty_cache()
        print(f"  [FullCT] {fid}: saved mask shape={binary.shape}, prob shape={prob.shape}")

    # ── Process normal findings (pipeline verifier) ──
    for i, (fid, prop_row) in enumerate(normal_findings):
        crop_info = finding_to_crop[fid]
        roi_path = Path(crop_info["cropped_roi_path"])
        gate_bbox = crop_info["gate_bbox_hwd"]

        if not roi_path.exists():
            print(f"  [SKIP] {fid}: cropped ROI missing: {roi_path}")
            continue

        # Load with NibabelIOWithReorient to match VoxTell training orientation
        reader = NibabelIOWithReorient()
        raw_nii = nib.load(str(roi_path))
        raw_shape = raw_nii.shape  # (H, W, D) — needed for bbox transform
        reoriented, _ = reader.read_images([str(roi_path)])
        cropped_roi = reoriented[0].astype(np.float32)  # (D, W, H) in reader space
        prompt = prop_row["prompt"]
        mask_out_dir = out_dir / "masks" / fid

        result = verifier.verify_finding(
            cropped_roi=cropped_roi,
            prompt=prompt,
            proposals=prop_row["proposals"],
            gate_bbox_hwd=gate_bbox,
            raw_shape=raw_shape,
            mask_out_dir=mask_out_dir,
        )

        verified_rows.append({
            "finding_id": fid,
            "prompt": prompt,
            "category": prop_row["category"],
            "verified": result["verified"],
            "rejected": result["rejected"],
            "fallback": result["fallback"],
            "gate_bbox_hwd": gate_bbox,
        })

        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            elapsed = time.time() - total_t0
            print(f"  [{i+1}/{len(findings)}] {elapsed:.0f}s")

    # Save
    out_file = out_dir / "verified_proposals.jsonl"
    if args.append_existing and out_file.exists():
        existing_rows = load_jsonl(out_file)
        merged_by_finding = {r["finding_id"]: r for r in existing_rows}
        for row in verified_rows:
            merged_by_finding[row["finding_id"]] = row
        existing_order = [r["finding_id"] for r in existing_rows]
        new_order = [r["finding_id"] for r in verified_rows
                     if r["finding_id"] not in set(existing_order)]
        verified_rows = [merged_by_finding[fid] for fid in existing_order + new_order]

    write_jsonl(out_file, verified_rows)

    # Summary
    n_verified = sum(len(r["verified"]) for r in verified_rows)
    n_rejected = sum(len(r["rejected"]) for r in verified_rows)
    n_fallback = sum(1 for r in verified_rows if r["fallback"])
    n_masks = sum(1 for r in verified_rows
                  for v in r["verified"]
                  if "coarse_mask_path" in v)

    summary = {
        "n_findings": len(verified_rows),
        "n_verified_proposals": n_verified,
        "n_rejected_proposals": n_rejected,
        "n_fallback_findings": n_fallback,
        "n_coarse_masks": n_masks,
        "total_time_s": round(time.time() - total_t0, 1),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nStage1 done: {len(verified_rows)} findings, "
          f"{n_verified} verified, {n_rejected} rejected, "
          f"{n_fallback} fallback")
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
