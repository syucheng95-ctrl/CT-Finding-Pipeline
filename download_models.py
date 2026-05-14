"""一键下载全部模型权重 (~10 GB total).

Usage:
  python download_models.py

Models:
  1. Qwen3-Embedding-4B (7.7 GB) → HuggingFace Hub
  2. TotalSegmentator (~300 MB) → 首次 import 自动下载
  3. VoxTell (1.7 GB) → HuggingFace Hub
  4. STU-Net (167 MB) → HuggingFace Hub
  5. Stage0 Router (9 MB) → HuggingFace Hub
  6. MONAI Nodule Detector (160 MB) → MONAI Model Zoo
"""

import os
import sys
import shutil
from pathlib import Path

MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_qwen():
    """Qwen3-Embedding-4B from HuggingFace Hub."""
    print("\n[1/5] Downloading Qwen3-Embedding-4B (7.7 GB)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "Qwen/Qwen3-Embedding-4B",
            local_dir=str(MODELS_DIR / "Qwen3-Embedding-4B"),
            local_dir_use_symlinks=False,
        )
        print("  [OK] Qwen3-Embedding-4B downloaded")
    except Exception as e:
        print(f"  [WARN] Qwen download failed: {e}")
        print("  You can manually download from: https://huggingface.co/Qwen/Qwen3-Embedding-4B")
        print(f"  Place files in: {MODELS_DIR / 'Qwen3-Embedding-4B'}")


def download_totalseg():
    """TotalSegmentator auto-downloads on first use."""
    print("\n[2/5] Checking TotalSegmentator...")
    try:
        import totalsegmentator
        print("  [OK] TotalSegmentator installed (will auto-download weights on first use)")
    except ImportError:
        print("  [WARN] pip install totalsegmentator first")
        print("  Weights auto-download on first run, no manual action needed")


def download_hub(repo_id: str, local_dir: str, name: str):
    """Download from HuggingFace Hub."""
    print(f"\n[{name}] Downloading {repo_id}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id, local_dir=str(MODELS_DIR / local_dir))
        print(f"  [OK] {name} downloaded")
    except Exception as e:
        print(f"  [WARN] {name} download failed: {e}")
        print(f"  Please manually download from: https://huggingface.co/{repo_id}")
        print(f"  Place files in: {MODELS_DIR / local_dir}")


def ensure_stage0_router_layout():
    """Normalize Stage0 router checkpoint to the path expected by config.yaml."""
    root = MODELS_DIR / "stage0_router"
    expected = root / "artifacts" / "models" / "router_head_best.pt"
    if expected.exists():
        print(f"  [OK] Stage0 Router checkpoint found: {expected}")
        return

    candidates = list(root.rglob("router_head_best.pt"))
    if not candidates:
        candidates = sorted(root.rglob("*.pt"))

    if candidates:
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], expected)
        print(f"  [OK] Copied Stage0 Router checkpoint to: {expected}")
        return

    raise FileNotFoundError(
        "Stage0 Router checkpoint not found after download. "
        f"Expected {expected}. Please check HuggingFace repo clover259/stage0-router-ct."
    )


def verify_required_files():
    """Fail early if required weights are missing."""
    required = [
        MODELS_DIR / "Qwen3-Embedding-4B",
        MODELS_DIR / "voxtell_v1.1",
        MODELS_DIR / "stage0_router" / "artifacts" / "models" / "router_head_best.pt",
        MODELS_DIR / "stunet" / "best.pt",
        MODELS_DIR / "lung_nodule_ct_detection",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n[ERROR] Missing required model files/directories:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)


def download_monai_nodule():
    """MONAI Lung Nodule Detection."""
    print("\n[5/5] Downloading MONAI Lung Nodule Detector (160 MB)...")
    try:
        from monai.bundle import load
        # This downloads and caches the bundle automatically
        _ = load(
            name="lung_nodule_ct_detection",
            bundle_dir=str(MODELS_DIR / "lung_nodule_ct_detection"),
        )
        print("  [OK] MONAI Nodule Detector downloaded")
    except Exception as e:
        print(f"  [WARN] MONAI download failed: {e}")
        print("  You can manually download with:")
        print("  python -c \"from monai.bundle import load; load('lung_nodule_ct_detection', bundle_dir='./models/lung_nodule_ct_detection')\"")


def main():
    print("=" * 60)
    print("Downloading all model weights for pipeline...")
    print("Total size: ~10 GB")
    print("=" * 60)

    # 1. Qwen3-Embedding-4B (7.7 GB, public via HuggingFace Hub)
    download_qwen()

    # 2. TotalSegmentator (auto-download, ~300 MB)
    download_totalseg()

    # 3. VoxTell checkpoint (你们训练的)
    download_hub(
        "clover259/Voxtell_v1.1",
        "voxtell_v1.1",
        "VoxTell (1.7 GB)"
    )

    # 4. STU-Net + MedIM (你们训练的)
    download_hub(
        "clover259/Stunet-ct",
        "stunet",
        "STU-Net (167 MB)"
    )

    # 5. Stage0 Router (你们训练的)
    download_hub(
        "clover259/stage0-router-ct",
        "stage0_router",
        "Stage0 Router (9 MB)"
    )
    ensure_stage0_router_layout()

    # 6. MONAI Nodule Detector
    download_monai_nodule()

    verify_required_files()

    print("\n" + "=" * 60)
    print("Done! Check ./models/ for downloaded weights.")
    print("If any step failed, check the manual download instructions above.")
    print("Then update config.yaml with the correct model paths.")
    print("=" * 60)


if __name__ == "__main__":
    main()
