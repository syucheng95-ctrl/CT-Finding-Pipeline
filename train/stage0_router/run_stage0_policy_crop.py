import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_qwen_embeddings import last_token_pool, pick_device, pick_dtype
from evaluate_stage0_policy_recall import (
    LungCache,
    bbox_for_mode,
    bbox_volume,
    policy_to_bbox,
)
from predict_router import load_head, predict_rows
from router_utils import write_jsonl


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def read_prompt_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"missing prompt in line {idx + 1}: {path}")
            row.setdefault("id", f"finding_{idx:04d}")
            rows.append(row)
    return rows


def parse_margin(text):
    vals = [float(x) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("margin must be H,W,D")
    return vals


def crop_nifti(nii, bbox):
    h0, h1, w0, w1, d0, d1 = bbox
    arr = np.asanyarray(nii.dataobj)
    cropped = arr[h0:h1, w0:w1, d0:d1]
    affine = nii.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(nii.affine, [h0, w0, d0])
    header = nii.header.copy()
    return nib.Nifti1Image(cropped, affine, header)


def load_router(checkpoint_path: Path, device):
    head, ckpt = load_head(checkpoint_path, device)
    cfg = ckpt["config"]
    qwen_cfg = cfg["qwen"]
    model_path = resolve_path(qwen_cfg["model_path"])
    dtype = pick_dtype(qwen_cfg.get("dtype", "auto"), device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    embedder = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    embedder.eval()
    return tokenizer, embedder, head, cfg


def main():
    parser = argparse.ArgumentParser(description="Run current Stage0 router + lungmask policy crops for one CT.")
    parser.add_argument("--image", required=True, help="Input CT .nii/.nii.gz")
    parser.add_argument("--prompts-jsonl", required=True, help="JSONL with at least {id,prompt}")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="router_training/artifacts/models/router_head_best.pt")
    parser.add_argument("--crop-mode", choices=["policy_group", "finding", "case_union"], default="policy_group")
    parser.add_argument("--mapping-mode", choices=["fixed", "auto"], default="fixed")
    parser.add_argument("--conservative-margin", default="60,60,50")
    parser.add_argument("--moderate-margin", default="40,40,30")
    parser.add_argument("--aggressive-margin", default="20,20,20")
    args = parser.parse_args()

    image_path = resolve_path(args.image)
    prompts_path = resolve_path(args.prompts_jsonl)
    out_dir = resolve_path(args.out_dir)
    crop_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = read_prompt_rows(prompts_path)
    if not rows:
        raise SystemExit(f"no prompts found: {prompts_path}")

    device = pick_device("auto")
    tokenizer, embedder, head, cfg = load_router(resolve_path(args.checkpoint), device)
    preds = predict_rows(rows, tokenizer, embedder, head, cfg, device)
    write_jsonl(preds, out_dir / "stage0_router_predictions.jsonl")

    nii = nib.load(str(image_path))
    shape = nii.shape
    zooms = nii.header.get_zooms()[:3]
    margins = {
        "conservative": parse_margin(args.conservative_margin),
        "moderate": parse_margin(args.moderate_margin),
        "aggressive": parse_margin(args.aggressive_margin),
    }

    lung_cache = LungCache()
    lung_info = lung_cache.get(image_path, shape, nii.affine, args.mapping_mode)

    prompt_bboxes = [
        policy_to_bbox(p["final_policy"], lung_info, shape, zooms, margins)
        for p in preds
    ]
    finding_bboxes, groups = bbox_for_mode(preds, prompt_bboxes, args.crop_mode)

    group_records = []
    stem = image_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem

    for group in groups:
        bbox = group["bbox"] or [0, shape[0], 0, shape[1], 0, shape[2]]
        group_id = group["group_id"].replace(":", "_").replace("/", "_")
        crop_name = f"{stem}__{group_id}.nii.gz"
        crop_path = crop_dir / crop_name
        nib.save(crop_nifti(nii, bbox), str(crop_path))

        group_prompt_ids = [
            p["id"]
            for p in preds
            if finding_bboxes.get(p["id"]) == bbox
        ]
        group_records.append(
            {
                "group_id": group["group_id"],
                "image": str(crop_path),
                "source_image": str(image_path),
                "bbox_hwd": bbox,
                "volume_ratio": bbox_volume(bbox) / max(1, int(np.prod(shape))),
                "n_findings": len(group_prompt_ids),
                "finding_ids": group_prompt_ids,
                "crop_mode": args.crop_mode,
                "mapping_mode": args.mapping_mode,
                "lungmask_transform": lung_info.get("transform"),
                "right_label": lung_info.get("right_label"),
                "left_label": lung_info.get("left_label"),
            }
        )

    write_jsonl(group_records, out_dir / "stage0_crop_groups.jsonl")
    summary = {
        "image": str(image_path),
        "prompts": str(prompts_path),
        "out_dir": str(out_dir),
        "crop_mode": args.crop_mode,
        "n_findings": len(preds),
        "n_crop_groups": len(group_records),
        "mean_group_volume_ratio": float(np.mean([r["volume_ratio"] for r in group_records])),
        "crop_groups": group_records,
    }
    with (out_dir / "stage0_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
