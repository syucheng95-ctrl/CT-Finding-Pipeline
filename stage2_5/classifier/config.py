from pathlib import Path

# ── Paths (all relative to upload/) ────────────────────────────────
UPLOAD_ROOT = Path(__file__).resolve().parent.parent.parent  # upload/

# Stage2 outputs (from pipeline)
ROI_ROOT = UPLOAD_ROOT / "outputs" / "stage2"
ROI_MANIFEST = ROI_ROOT / "roi_manifest.jsonl"
ROI_IMAGE_DIR = ROI_ROOT / "roi_images"

# Model weights (from download_models.py)
MEDIM_CKPT_DIR = UPLOAD_ROOT / "models" / "medim_ckpt"
QWEN_MODEL_DIR = UPLOAD_ROOT / "models" / "Qwen3-Embedding-4B"

OUTPUT_ROOT = UPLOAD_ROOT / "stage2_5" / "outputs"
CHECKPOINT_DIR = OUTPUT_ROOT / "classifier_checkpoints"

# ── Model ──────────────────────────────────────────────────────────
MEDIM_VARIANT = "STU-Net-S"
MEDIM_DATASET = "TotalSegmentator"
MEDIM_ENCODER_CH = 256          # bottleneck channels
IMAGE_PROJ_DIM = 512            # project MedIM 256d → 512d
QWEN_EMBED_DIM = 2560           # Qwen3-Embedding-4B hidden_size
FUSION_HIDDEN_DIM = 96          # FC hidden after concat
DROPOUT = 0.4

# ── Data ───────────────────────────────────────────────────────────
PATCH_SIZE = (96, 96, 48)       # fixed input size (x, y, z)
CT_CLIP_LO = -1000.0
CT_CLIP_HI = 400.0
NUM_WORKERS = 0                 # DataLoader workers (0 = main process only, avoids Windows spawn issues)

# ── Training ───────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 25
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LABEL_SMOOTHING = 0.05

# ── Data split ──────────────────────────────────────────────────────
VAL_SPLIT_RATIO = 0.20       # 80/20 stratified case-level split
SPLIT_SEED = 42              # fixed seed for reproducibility

# ── Focal Loss ───────────────────────────────────────────────────────
FOCAL_GAMMA = 2.0            # focusing parameter (higher = more weight on hard samples)

# ── 14 lesion categories → 5 metric groups ─────────────────────────
CATEGORY_LIST = [
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d",
    "2e", "2f", "2g", "2h",
]

CATEGORY_TO_METRIC_GROUP = {
    # lung_nodule
    "1a": "lung_nodule", "1b": "lung_nodule", "1c": "lung_nodule",
    "1d": "lung_nodule", "1e": "lung_nodule", "1f": "lung_nodule",
    "2a": "lung_nodule", "2b": "lung_nodule", "2c": "lung_nodule", "2d": "lung_nodule",
    # lung_mass
    "2e": "lung_mass", "2f": "lung_mass", "2g": "lung_mass", "2h": "lung_mass",
}

CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORY_LIST)}
IDX_TO_CATEGORY = {i: cat for cat, i in CATEGORY_TO_IDX.items()}
NUM_CLASSES = len(CATEGORY_LIST)
