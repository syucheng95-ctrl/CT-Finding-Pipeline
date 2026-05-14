"""Compute Dice score for Stage0.5 proposals vs GT labels."""
import json
from collections import defaultdict
from pathlib import Path
import nibabel as nib
import numpy as np

props_path = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage0.5_workspace\outputs\val20_full_v2\stage0_5_proposals.jsonl")

# Read manifest directly (plain JSONL)
demo = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\生医工大赛-demo")
manifest_path = list(demo.rglob("finding_mask_manifest.jsonl"))[0]
manifest = {}
with manifest_path.open(encoding="utf-8") as mf:
    for line in mf:
        line = line.strip()
        if line:
            r = json.loads(line)
            manifest[r["id"]] = r

dices = []
by_cat = defaultdict(list)

with props_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fid = r["finding_id"]
        m = manifest.get(fid)
        if m is None:
            continue

        label_nii = nib.load(str(m["label"]))
        label = np.asanyarray(label_nii.dataobj) > 0
        gt_total = int(label.sum())

        pred = np.zeros(label.shape, dtype=bool)
        for p in r["proposals"]:
            b = p["proposal_bbox_hwd"]
            H, W, D = label.shape
            h0, h1 = max(0, b[0]), min(H, b[1])
            w0, w1 = max(0, b[2]), min(W, b[3])
            d0, d1 = max(0, b[4]), min(D, b[5])
            if h0 < h1 and w0 < w1 and d0 < d1:
                pred[h0:h1, w0:w1, d0:d1] = True

        pred_total = int(pred.sum())
        if pred_total == 0 and gt_total == 0:
            dice = 1.0
        elif pred_total == 0 or gt_total == 0:
            dice = 0.0
        else:
            inter = int((pred & label).sum())
            dice = 2 * inter / (pred_total + gt_total)

        dices.append(dice)
        by_cat[r["category"]].append(dice)

dices = np.array(dices)
print(f"\n{'='*50}")
print(f"N={len(dices)} findings")
print(f"Mean Dice:   {dices.mean():.4f}")
print(f"Median Dice: {np.median(dices):.4f}")
print(f"P05 Dice:    {np.percentile(dices, 5):.4f}")
print(f"P95 Dice:    {np.percentile(dices, 95):.4f}")
print(f"Dice > 0:    {(dices > 0).mean():.1%}")

print(f"\n{'='*50}")
print("Per-category Dice:")
names = {
    "1a": "bronchial_wall", "1b": "bronchiectasis", "1c": "emphysema",
    "1d": "septal_thick", "1e": "micronodule", "1f": "other_non_focal",
    "2a": "linear_scar", "2b": "consolidation", "2c": "GGO",
    "2d": "nodule_mass", "2f": "honeycombing", "2g": "pneumothorax",
}
for cat in sorted(by_cat.keys()):
    arr = np.array(by_cat[cat])
    name = names.get(cat, cat)
    print(f"  {cat} {name:20s}: n={len(arr):2d}  mean_dice={arr.mean():.4f}  median={np.median(arr):.4f}  min={arr.min():.4f}")
