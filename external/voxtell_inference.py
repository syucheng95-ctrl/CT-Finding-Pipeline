"""Adapter for VoxTell inference (used by Stage1 verification).

Provides: build_voxtell_model(model_dir, device) → model
          NibabelIOWithReorient (image I/O matching VoxTell training orientation)
"""

import os
import sys
from pathlib import Path


def ensure_voxtell_in_path():
    """Add voxtell-related paths to sys.path."""
    upload_dir = Path(__file__).resolve().parent.parent
    # VoxTell wheel now lives in upload/workspace/voxtell/
    vox_path = str(upload_dir / "workspace" / "voxtell")
    if vox_path not in sys.path:
        sys.path.insert(0, vox_path)
    # Stage1 workspace now lives in upload/workspace/stage1/
    s1_path = str(upload_dir / "workspace" / "stage1")
    if s1_path not in sys.path:
        sys.path.insert(0, s1_path)


def get_nibabel_io_with_reorient():
    """Get NibabelIOWithReorient for VoxTell-compatible image loading."""
    ensure_voxtell_in_path()
    try:
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        return NibabelIOWithReorient
    except ImportError:
        raise ImportError(
            "VoxTell not found. Set VOXTELL_PATH and STAGE1_PATH env vars, "
            "or place 生医工大赛-demo/ and stage1_verification/ next to upload/."
        )


def build_voxtell_model(model_dir: str, device: str = "cuda:0"):
    """Load VoxTell model from checkpoint."""
    ensure_voxtell_in_path()
    try:
        from config import build_voxtell_from_checkpoint
        import torch
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = build_voxtell_from_checkpoint(Path(model_dir), device=torch.device("cpu"))
        return model.to(device).eval()
    except ImportError:
        raise ImportError(
            "VoxTell config/build not found. "
            "Ensure stage1_verification/ is accessible (set STAGE1_PATH env var)."
        )
