"""Adapter for STU-Net inference (used by Stage2 evaluation).

Provides: load_stunet_model(checkpoint_path, device) → (model, patch_shapes)
          get_group, load_group_patch_shapes, pad_to_shape_centered, prepare_image
"""

import os
import sys
from pathlib import Path


def ensure_stunet_in_path():
    """Add workspace/stage2 to sys.path."""
    upload_dir = Path(__file__).resolve().parent.parent
    ws = str(upload_dir / "workspace" / "stage2")
    if ws not in sys.path:
        sys.path.insert(0, ws)
    return ws


def load_stunet_model(checkpoint_path: str, device: str = "cuda"):
    """Load STU-Net model and return (model, patch_shapes)."""
    ws = ensure_stunet_in_path()
    if not ws:
        raise FileNotFoundError(
            "stage2_workspace not found. "
            "Set STAGE2_WORKSPACE env var or place stage2_workspace next to upload/."
        )

    # Set medim checkpoint dir (look in upload/models/ first, then workspace)
    upload_medim = Path(__file__).resolve().parent.parent / "models" / "medim_ckpt"
    if upload_medim.exists():
        os.environ["MEDIM_CKPT_DIR"] = str(upload_medim)
    else:
        os.environ.setdefault("MEDIM_CKPT_DIR",
                              str(Path(ws) / "data" / "medim_ckpt"))

    # Force offline mode — pretrained weights are bundled in upload/models/
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from eval_stratified import get_group, load_group_patch_shapes, pad_to_shape_centered, prepare_image
    from model import create_stunet_model

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"STU-Net checkpoint not found: {ckpt_path}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = create_stunet_model(variant="STU-Net-S", pretrained_dataset=None,
                                out_channels=2)
    ckpt_state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_state["model_state_dict"])
    model = model.to(device).eval()
    gps = load_group_patch_shapes(ckpt_state)

    return model, gps, (get_group, pad_to_shape_centered, prepare_image)
