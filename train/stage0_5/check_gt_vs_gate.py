"""Check if GT label falls inside gate bbox for failed nodule findings."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import find_objects
import sys

STAGE0_V2 = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage0_workspace\stage0_v2\router")
if str(STAGE0_V2) not in sys.path:
    sys.path.insert(0, str(STAGE0_V2))
from router_utils import read_jsonl

failed = [
    "finding_004747", "finding_001004", "finding_005559",
    "finding_005568", "finding_001910", "finding_007426",
]

# Use same path logic as run_val20.py
proposals_path = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage0.5_workspace\outputs\val20_full\stage0_5_proposals.jsonl")

# Find manifest from the known location
manifest_path = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\生医工大赛-demo\full_dataset_work\clean_full\manifests\finding_mask_manifest.jsonl")
if not manifest_path.exists():
    # fallback: search
    demo = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\生医工大赛-demo")
    for p in demo.rglob("finding_mask_manifest.jsonl"):
        manifest_path = p
        break

manifest = {r["id"]: r for r in read_jsonl(manifest_path)}

props = {}
with proposals_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            props[r["finding_id"]] = r

for fid in failed:
    p = props.get(fid)
    m = manifest.get(fid)
    if p is None or m is None:
        print(f"{fid}: missing data (props={p is not None}, manifest={m is not None})")
        continue

    label_path = m["label"]
    if not Path(label_path).exists():
        print(f"{fid}: label not found: {label_path}")
        continue

    label_nii = nib.load(str(label_path))
    label_mask = (np.asanyarray(label_nii.dataobj) > 0).astype(np.uint8)
    gt_voxels = int(label_mask.sum())

    slices = find_objects(label_mask)
    gt_bbox = None
    if slices:
        for s in slices:
            if s is not None:
                gt_bbox = [s[0].start, s[0].stop, s[1].start, s[1].stop, s[2].start, s[2].stop]
                break

    gate = p["gate_bbox_hwd"]
    print(f"\n{fid} cat={p['category']} gt_voxels={gt_voxels}")

    if gt_bbox is None:
        print(f"  GT has no positive voxels")
        continue

    inside = (
        gt_bbox[0] >= gate[0]
        and gt_bbox[1] <= gate[1]
        and gt_bbox[2] >= gate[2]
        and gt_bbox[3] <= gate[3]
        and gt_bbox[4] >= gate[4]
        and gt_bbox[5] <= gate[5]
    )

    gt_center = [
        (gt_bbox[0] + gt_bbox[1]) // 2,
        (gt_bbox[2] + gt_bbox[3]) // 2,
        (gt_bbox[4] + gt_bbox[5]) // 2,
    ]
    gate_center = [
        (gate[0] + gate[1]) // 2,
        (gate[2] + gate[3]) // 2,
        (gate[4] + gate[5]) // 2,
    ]
    dist = [abs(gt_center[i] - gate_center[i]) for i in range(3)]

    gt_size = [gt_bbox[1] - gt_bbox[0], gt_bbox[3] - gt_bbox[2], gt_bbox[5] - gt_bbox[4]]
    gate_size = [gate[1] - gate[0], gate[3] - gate[2], gate[5] - gate[4]]

    print(f"  gate: {gate}  size: {gate_size}")
    print(f"  GT:   {gt_bbox}  size: {gt_size}")
    print(f"  GT inside gate: {inside}")
    print(f"  gate center: {gate_center}, GT center: {gt_center}, dist: {dist}")

    # Check best IoU
    best_iou = 0.0
    best_prop_id = None
    for prop in p.get("proposals", []):
        pb = prop["proposal_bbox_hwd"]
        h0, h1 = max(pb[0], gt_bbox[0]), min(pb[1], gt_bbox[1])
        w0, w1 = max(pb[2], gt_bbox[2]), min(pb[3], gt_bbox[3])
        d0, d1 = max(pb[4], gt_bbox[4]), min(pb[5], gt_bbox[5])
        inter = max(0, h1 - h0) * max(0, w1 - w0) * max(0, d1 - d0)
        vol_p = (pb[1] - pb[0]) * (pb[3] - pb[2]) * (pb[5] - pb[4])
        vol_g = (gt_bbox[1] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[2]) * (gt_bbox[5] - gt_bbox[4])
        union = vol_p + vol_g - inter
        iou = inter / max(1, union)
        if iou > best_iou:
            best_iou = iou
            best_prop_id = prop.get("proposal_id", "?")
    print(f"  Best proposal IoU with GT: {best_iou:.4f} ({best_prop_id})")
