"""Check if failed nodule findings have any raw detector hits (before thresholding)."""
import sys
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "stage0_workspace/stage0_v2/router"))

from router_utils import read_jsonl

# Add proposal_generator to path
sys.path.insert(0, str(Path(__file__).parent))

from proposal_generator.nodule_detector_expert import run_nodule_detector_expert, _RAW_DETECTION_CACHE
from proposal_generator.input_parser import build_finding_to_group_index

# Replicate the val20 detection flow
failed_ids = ["finding_004747", "finding_001004", "finding_005559", "finding_005568", "finding_001910", "finding_007426"]

manifest = {r["id"]: r for r in read_jsonl(Path(
    r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛"
    r"\生医工大赛-demo\full_dataset_work\clean_full\manifests"
    r"\finding_mask_manifest.jsonl"
))}

for fid in failed_ids:
    rec = manifest.get(fid)
    if rec is None:
        print(f"{fid}: NOT IN MANIFEST")
        continue
    case_name = rec["case_name"]
    ct_path = None
    for base in [
        r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage1_verification\data\images",
        r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage1_verification\data\images_flat",
    ]:
        p = Path(base) / case_name
        if p.exists():
            ct_path = p
            break
    if ct_path is None:
        print(f"{fid}: CT NOT FOUND")
        continue

    # For this test, just use a large gate bbox covering most of the lung
    nii = nib.load(str(ct_path))
    shape = nii.shape
    zooms = nii.header.get_zooms()[:3]

    # Check detection at different score thresholds
    print(f"\n=== {fid} ({case_name}) ===")
    print(f"  shape={shape}  spacing={zooms[0]:.3f}x{zooms[1]:.3f}x{zooms[2]:.3f}")

    # Try with NO score threshold to see all raw detections
    raw = run_nodule_detector_expert(nii, "1e", cache_key=None)

    if raw is None:
        print(f"  ZERO raw detections (before any threshold)")
    else:
        print(f"  {len(raw)} raw detections, top scores: {[f'{d[\"score\"]:.4f}' for d in sorted(raw, key=lambda x: x['score'], reverse=True)[:10]]}")

    # Also check 2d threshold
    raw2d = run_nodule_detector_expert(nii, "2d", cache_key=None)
    if raw2d is None or len(raw2d) == 0:
        print(f"  2d threshold (0.05): 0 detections")
    else:
        print(f"  2d threshold (0.05): {len(raw2d)} detections, top scores: {[f'{d[\"score\"]:.4f}' for d in sorted(raw2d, key=lambda x: x['score'], reverse=True)[:5]]}")

    # Clear cache for next finding
    _RAW_DETECTION_CACHE.clear()
