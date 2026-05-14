"""Evaluate VoxTell prediction vs GT for finding_004746."""
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

VOXTELL = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\生医工大赛-demo\third_party\voxtell_pkg\voxtell_wheel")
sys.path.insert(0, str(VOXTELL.resolve()))
from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
import torch

MODEL_DIR = str((Path(__file__).resolve().parent.parent / "生医工大赛-demo/data/voxtell_ckpt/voxtell_v1.1").resolve())
TEXT_MODEL = str((Path(__file__).resolve().parent.parent / "生医工大赛-demo/data/modelscope_models/Qwen/Qwen3-Embedding-4B").resolve())
CT_PATH = str((Path(__file__).resolve().parent.parent / "stage1_verification/data/images/train_11880_d_1.nii.gz").resolve())

# Load manifest to get GT label path
STAGE0_V2 = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\stage0_workspace\stage0_v2\router")
sys.path.insert(0, str(STAGE0_V2))
from router_utils import read_jsonl

demo = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\生医工大赛-demo")
manifest_path = list(demo.rglob("finding_mask_manifest.jsonl"))[0]
manifest = {}
with manifest_path.open(encoding="utf-8") as mf:
    for line in mf:
        line = line.strip()
        if not line:
            continue
        r = __import__('json').loads(line)
        manifest[r["id"]] = r

# finding_004746 is the consolidation on train_11880_d_1
fid = "finding_004746"
rec = manifest[fid]
PROMPT = rec["prompt"]
label_path = rec["label"]
print(f"Finding: {fid}")
print(f"Prompt: {PROMPT}")
print(f"Label: {label_path}")

# Load predictor
print("\nLoading VoxTell...")
predictor = VoxTellPredictor(
    model_dir=MODEL_DIR,
    device=torch.device("cuda:0"),
    text_encoding_model=TEXT_MODEL,
)

reader = NibabelIOWithReorient()
img, props = reader.read_images([CT_PATH])
img_array = img[0]

print(f"Running inference...")
pred = predictor.predict_single_image(img_array, [PROMPT])
pred_bin = pred[0] > 0  # (D, H, W)

# Load GT
gt_nii = nib.load(str(label_path))
gt = np.asanyarray(gt_nii.dataobj) > 0

print(f"GT shape: {gt.shape}, Pred shape: {pred_bin.shape}")

# GT is (H,W,D), pred is (D,H,W) — transpose GT to match
gt = np.transpose(gt, (2, 0, 1))  # now (D,H,W)
print(f"After transpose — GT: {gt.shape}, Pred: {pred_bin.shape}")

# Compute Dice and Recall
inter = int((pred_bin & gt).sum())
pred_total = int(pred_bin.sum())
gt_total = int(gt.sum())
dice = 2 * inter / max(1, pred_total + gt_total)
recall = inter / max(1, gt_total)
precision = inter / max(1, pred_total)

print(f"\n=== VoxTell Full-Volume Results ===")
print(f"GT voxels:     {gt_total}")
print(f"Pred voxels:   {pred_total}")
print(f"Intersection:  {inter}")
print(f"Dice:          {dice:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"Precision:     {precision:.4f}")
