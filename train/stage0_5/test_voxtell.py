"""Quick VoxTell test on one val20 CT with one finding prompt."""
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

VOXTELL = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\生医工大赛-demo\third_party\voxtell_pkg\voxtell_wheel")
sys.path.insert(0, str(VOXTELL.resolve()))
from voxtell.inference.predictor import VoxTellPredictor

MODEL_DIR = str((Path(__file__).resolve().parent.parent / "生医工大赛-demo/data/voxtell_ckpt/voxtell_v1.1").resolve())
TEXT_MODEL = str((Path(__file__).resolve().parent.parent / "生医工大赛-demo/data/modelscope_models/Qwen/Qwen3-Embedding-4B").resolve())
CT_PATH = str((Path(__file__).resolve().parent.parent / "stage1_verification/data/images/train_11880_d_1.nii.gz").resolve())

# Use a finding prompt from val20
PROMPT = "Consolidation with air bronchogram in the lower lobe of the right lung with surrounding ground-glass opacity"

print("Loading VoxTell predictor...")
predictor = VoxTellPredictor(
    model_dir=MODEL_DIR,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    text_encoding_model=TEXT_MODEL,
)
print("Done.")

print(f"Loading CT: {CT_PATH}")
reader = NibabelIOWithReorient()
img, props = reader.read_images([CT_PATH])
img_array = img[0]  # (1, X, Y, Z)
print(f"CT shape: {img_array.shape}")

print(f"Running VoxTell on: {PROMPT[:80]}...")
pred = predictor.predict_single_image(img_array, [PROMPT])
print(f"Prediction shape: {pred.shape}")  # (1, D, H, W)
print(f"Foreground voxels: {pred.sum()} / {pred.size}")
print(f"Activation ratio: {pred.sum() / pred.size:.6f}")

# Quick check: did it find something?
fg = int(pred.sum())
print(f"\nVoxTell found {fg} foreground voxels in full volume.")
if fg > 0:
    print("VoxTell DID activate on this finding.")
else:
    print("VoxTell found NOTHING.")
