"""Check if MONAI detector can see GT nodules with minimal thresholding."""
import sys
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import find_objects, zoom

STAGE0_V2 = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage0_workspace\stage0_v2\router")
if str(STAGE0_V2) not in sys.path:
    sys.path.insert(0, str(STAGE0_V2))
from router_utils import read_jsonl

# Import detector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposal_generator.nodule_detector_expert import (
    _load_detector, _preprocess_hu, _box_to_hwd, _RAW_DETECTION_CACHE,
    TARGET_SPACING, NODULE_DETECTOR_MAX_DETECTIONS,
)

# Find the failed cases from val20 output
props_path = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage0.5_workspace\outputs\val20_full\stage0_5_proposals.jsonl")
props = {}
with props_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            props[r["finding_id"]] = r

failed = ["finding_004747", "finding_001004", "finding_005559"]

# Find manifest
demo = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\生医工大赛-demo")
manifest = {r["id"]: r for r in read_jsonl(list(demo.rglob("finding_mask_manifest.jsonl"))[0])}

# Find CT images
image_dirs = [
    Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage1_verification\data\images"),
    Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage1_verification\data\images_flat"),
]

# Load detector once
loaded = _load_detector()
detector = loaded["detector"]
device = loaded["device"]

# Temporarily lower the internal score threshold
detector.set_box_selector_parameters(
    score_thresh=0.001,  # near-zero to see everything
    topk_candidates_per_level=2000,
    nms_thresh=0.22,
    detections_per_img=200,
)

for fid in failed:
    p = props.get(fid)
    m = manifest.get(fid)
    if p is None or m is None:
        continue

    case_name = m["case_name"]
    ct_path = None
    for d in image_dirs:
        cp = d / case_name
        if cp.exists():
            ct_path = cp
            break
    if ct_path is None:
        continue

    nii = nib.load(str(ct_path))
    gate = p["gate_bbox_hwd"]
    h0, h1, w0, w1, d0, d1 = gate
    # Safety clamp
    H, W, D = nii.shape
    h0, h1 = max(0, h0), min(H, h1)
    w0, w1 = max(0, w0), min(W, w1)
    d0, d1 = max(0, d0), min(D, d1)

    arr = np.asanyarray(nii.dataobj)[h0:h1, w0:w1, d0:d1].copy().astype(np.float32)
    affine = nii.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(nii.affine, [h0, w0, d0])
    cropped = nib.Nifti1Image(arr, affine, nii.header)

    # Run with near-zero threshold
    image, scale = _preprocess_hu(cropped)
    image = image.to(device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            pred = detector([image], use_inferer=False)[0]

    boxes = pred.get("box")
    scores = pred.get("label_scores")

    # Load GT
    label_nii = nib.load(str(m["label"]))
    label_mask = (np.asanyarray(label_nii.dataobj) > 0).astype(np.uint8)
    gt_voxels = int(label_mask.sum())
    objects = find_objects(label_mask)
    gt_bbox = None
    if objects:
        for s in objects:
            if s is not None:
                gt_bbox = [s[0].start, s[0].stop, s[1].start, s[1].stop, s[2].start, s[2].stop]
                break

    print(f"\n=== {fid} cat={p['category']} gt_voxels={gt_voxels} ===")
    print(f"  CT spacing: {[f'{z:.3f}' for z in cropped.header.get_zooms()[:3]]}")
    print(f"  Crop shape: {cropped.shape}")
    if gt_bbox:
        print(f"  GT bbox (original): {gt_bbox}")
        print(f"  GT bbox in crop: {[gt_bbox[0]-gate[0], gt_bbox[1]-gate[0], gt_bbox[2]-gate[2], gt_bbox[3]-gate[2], gt_bbox[4]-gate[4], gt_bbox[5]-gate[4]]}")

    if boxes is None or scores is None or len(boxes) == 0:
        print(f"  ZERO detections even with threshold=0.001!")
        continue

    boxes_np = boxes.float().cpu().numpy()
    scores_np = scores.float().cpu().numpy()

    print(f"  Total raw detections: {len(boxes_np)}")
    print(f"  Score range: {scores_np.min():.6f} ~ {scores_np.max():.6f}")

    # Check each detection against GT
    best_iou = 0.0
    best_info = None
    all_dets = []
    for i, (box, score) in enumerate(zip(boxes_np, scores_np)):
        bbox = _box_to_hwd(box, cropped.shape, scale)
        if bbox is None:
            continue
        # Map to original coordinates
        bbox_orig = [bbox[0] + gate[0], bbox[1] + gate[0],
                     bbox[2] + gate[2], bbox[3] + gate[2],
                     bbox[4] + gate[4], bbox[5] + gate[4]]
        all_dets.append((score, bbox_orig))

        if gt_bbox:
            # Compute IoU
            oh0, oh1 = max(bbox_orig[0], gt_bbox[0]), min(bbox_orig[1], gt_bbox[1])
            ow0, ow1 = max(bbox_orig[2], gt_bbox[2]), min(bbox_orig[3], gt_bbox[3])
            od0, od1 = max(bbox_orig[4], gt_bbox[4]), min(bbox_orig[5], gt_bbox[5])
            inter = max(0, oh1 - oh0) * max(0, ow1 - ow0) * max(0, od1 - od0)
            vol_b = (bbox_orig[1] - bbox_orig[0]) * (bbox_orig[3] - bbox_orig[2]) * (bbox_orig[5] - bbox_orig[4])
            vol_g = max(1, (gt_bbox[1] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[2]) * (gt_bbox[5] - gt_bbox[4]))
            iou = inter / (vol_b + vol_g - inter)
            if iou > best_iou:
                best_iou = iou
                best_info = (i, float(score), bbox_orig, [bbox_orig[1]-bbox_orig[0], bbox_orig[3]-bbox_orig[2], bbox_orig[5]-bbox_orig[4]])

    print(f"  Valid detections: {len(all_dets)}")
    if best_info:
        print(f"  Best IoU with GT: {best_iou:.6f} (det #{best_info[0]}, score={best_info[1]:.6f}, size={best_info[3]})")

    # Top 20 detections by score
    all_dets.sort(key=lambda x: -x[0])
    print(f"  Top 15 scores: {[f'{s:.4f}' for s, _ in all_dets[:15]]}")

    if all_dets:
        # Also show detections near GT center
        if gt_bbox:
            gc = [(gt_bbox[0]+gt_bbox[1])//2, (gt_bbox[2]+gt_bbox[3])//2, (gt_bbox[4]+gt_bbox[5])//2]
            print(f"  GT center: {gc}")
            # Find detections close to GT
            close = []
            for score, bbox in all_dets:
                bc = [(bbox[0]+bbox[1])//2, (bbox[2]+bbox[3])//2, (bbox[4]+bbox[5])//2]
                dist = np.sqrt(sum((gc[i] - bc[i])**2 for i in range(3)))
                if dist < 50:  # within 50 voxels
                    close.append((dist, score, bbox))
            close.sort(key=lambda x: x[0])
            if close:
                print(f"  Detections within 50 voxels of GT center:")
                for dist, score, bbox in close[:5]:
                    print(f"    dist={dist:.1f} score={score:.4f} bbox={bbox}")
            else:
                print(f"  NO detections within 50 voxels of GT center!")
