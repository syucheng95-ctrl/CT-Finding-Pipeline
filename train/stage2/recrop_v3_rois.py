import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


GROUP_ORDER = ["small", "medium", "large", "xlarge"]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def get_group(hwd):
    h, w, d = [int(v) for v in hwd]
    if h <= 64 and w <= 64 and d <= 32:
        return "small"
    if d > 48:
        return "xlarge"
    if h > 96 or w > 96:
        return "large"
    return "medium"


def component_key(row):
    for key in ("component_sample_id", "component_id"):
        if row.get(key):
            return str(row[key])
    roi_id = str(row.get("roi_id", ""))
    if "_comp_" in roi_id:
        return "comp_" + roi_id.rsplit("_comp_", 1)[1]
    return roi_id


def hwd_from_zyx_box(box_zyx):
    z1, h1, w1, z2, h2, w2 = [int(v) for v in box_zyx]
    return [h1, w1, z1, h2, w2, z2]


def resolve_path(root: Path, rel: str):
    path = Path(rel)
    if path.is_absolute():
        return path
    candidates = [root / path, root / "clean_full" / path, root / "stage2_roi" / path]
    candidates.append(root / "ct_cache" / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def bbox_from_roi_meta(meta):
    if meta.get("component_box_hwd"):
        return [int(v) for v in meta["component_box_hwd"]]
    if meta.get("component_box_zyx"):
        return hwd_from_zyx_box(meta["component_box_zyx"])
    raise KeyError("roi metadata is missing component_box_hwd/component_box_zyx")


def proportional_margin(box_hwd, min_margin_hwd, ratio):
    h1, w1, d1, h2, w2, d2 = box_hwd
    lesion_hwd = (h2 - h1, w2 - w1, d2 - d1)
    return tuple(max(int(min_margin_hwd[i]), int(round(lesion_hwd[i] * ratio))) for i in range(3))


def expand_box(box_hwd, shape_hwd, margin_hwd):
    h1, w1, d1, h2, w2, d2 = box_hwd
    mh, mw, md = margin_hwd
    return [
        max(0, h1 - mh),
        max(0, w1 - mw),
        max(0, d1 - md),
        min(shape_hwd[0], h2 + mh),
        min(shape_hwd[1], w2 + mw),
        min(shape_hwd[2], d2 + md),
    ]


def crop(volume, box_hwd):
    h1, w1, d1, h2, w2, d2 = box_hwd
    return volume[h1:h2, w1:w2, d1:d2]


def crop_slices(box_hwd):
    h1, w1, d1, h2, w2, d2 = box_hwd
    return np.s_[h1:h2, w1:w2, d1:d2]


def save_nifti_like(ref: nib.Nifti1Image, arr: np.ndarray, path: Path, dtype=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is not None:
        arr = arr.astype(dtype)
    out = nib.Nifti1Image(arr, affine=ref.affine, header=ref.header.copy())
    if dtype is not None:
        out.header.set_data_dtype(dtype)
    nib.save(out, str(path))


def copy_existing_roi(row, old_roi_root: Path, out_root: Path):
    image_rel = row.get("roi_image_path") or row.get("roi_image")
    mask_rel = row.get("roi_mask_path") or row.get("roi_mask")
    img_src = resolve_path(old_roi_root, image_rel)
    mask_src = resolve_path(old_roi_root, mask_rel)
    img_dst = out_root / image_rel
    mask_dst = out_root / mask_rel
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    if not img_dst.exists():
        shutil.copy2(img_src, img_dst)
    if not mask_dst.exists():
        shutil.copy2(mask_src, mask_dst)


def recrop_one(row, meta, clean_root: Path, out_root: Path, ratio: float, min_margin_hwd):
    source_image = row.get("source_image") or meta.get("source_image")
    source_label = row.get("source_label") or meta.get("parent_label")
    if source_image is None or source_label is None:
        raise KeyError("missing source_image/source_label or parent_label")

    image_path = resolve_path(clean_root, source_image)
    label_path = resolve_path(clean_root, source_label)
    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    image = np.asanyarray(image_nii.dataobj)
    label = np.asanyarray(label_nii.dataobj)

    if image.ndim == 4:
        image = image[..., 0]
    if label.ndim == 4:
        label = label[..., 0]
    if image.shape != label.shape:
        raise ValueError(f"image shape {image.shape} != label shape {label.shape}")

    comp_box = bbox_from_roi_meta(meta)
    margin_hwd = proportional_margin(comp_box, min_margin_hwd, ratio)
    roi_box = expand_box(comp_box, image.shape, margin_hwd)

    comp_mask = np.zeros(label.shape, dtype=np.uint8)
    comp_mask[crop_slices(comp_box)] = (crop(label, comp_box) > 0).astype(np.uint8)
    roi_image = crop(image, roi_box)
    roi_mask = crop(comp_mask, roi_box)

    image_rel = row.get("roi_image_path") or row.get("roi_image")
    mask_rel = row.get("roi_mask_path") or row.get("roi_mask")
    save_nifti_like(image_nii, roi_image.astype(np.float32, copy=False), out_root / image_rel, dtype=np.float32)
    save_nifti_like(label_nii, roi_mask, out_root / mask_rel, dtype=np.uint8)

    new_row = dict(row)
    new_row.setdefault("component_sample_id", component_key(row))
    new_row.setdefault("parent_sample_id", row.get("parent_id") or meta.get("parent_id"))
    new_row.setdefault("roi_image_path", image_rel)
    new_row.setdefault("roi_mask_path", mask_rel)
    new_row["source_image"] = source_image
    new_row["source_label"] = source_label
    new_row["array_axis_order"] = "xyz"
    new_row["source_image_shape_xyz"] = [int(v) for v in image.shape]
    new_row["roi_shape_hwd"] = [int(v) for v in roi_image.shape]
    new_row["voxel_count"] = int(np.count_nonzero(roi_mask))
    new_row["v3_recrop"] = True
    new_row["v3_component_box_hwd"] = comp_box
    new_row["v3_roi_box_hwd"] = roi_box
    new_row["v3_margin_hwd"] = [int(v) for v in margin_hwd]
    new_row["v3_margin_ratio"] = ratio
    return new_row


def main():
    parser = argparse.ArgumentParser(description="Recrop large/xlarge Stage 2 ROIs with proportional margins for V3.")
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest with final train/val/holdout split.")
    parser.add_argument("--roi-manifest", type=Path, required=True, help="Original stage2-roi roi_manifest.jsonl with component boxes.")
    parser.add_argument("--old-roi-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, default=None)
    parser.add_argument("--groups", nargs="+", default=["large", "xlarge"], choices=GROUP_ORDER)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--min-margin-hwd", nargs=3, type=int, default=[16, 16, 16])
    parser.add_argument("--copy-unchanged", action="store_true", help="Copy small/medium ROI files into out-root too.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.out_manifest is None:
        args.out_manifest = args.out_root / "manifest_v3.jsonl"

    rows = load_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    roi_meta = {component_key(row): row for row in load_jsonl(args.roi_manifest)}

    out_rows = []
    stats = Counter()
    errors = []

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "roi_images").mkdir(parents=True, exist_ok=True)
    (args.out_root / "roi_masks").mkdir(parents=True, exist_ok=True)

    target_groups = set(args.groups)
    for row in tqdm(rows, desc="recrop V3 ROI"):
        group = get_group(row["roi_shape_hwd"])
        key = component_key(row)
        try:
            if group in target_groups:
                if key not in roi_meta:
                    raise KeyError(f"component {key} not found in roi manifest")
                new_row = recrop_one(row, roi_meta[key], args.clean_root, args.out_root, args.ratio, tuple(args.min_margin_hwd))
                stats[f"recrop_{group}"] += 1
            else:
                new_row = dict(row)
                new_row["v3_recrop"] = False
                if args.copy_unchanged:
                    copy_existing_roi(row, args.old_roi_root, args.out_root)
                stats[f"keep_{group}"] += 1
            out_rows.append(new_row)
        except Exception as exc:
            errors.append({"id": key, "group": group, "error": repr(exc)})
            stats["error"] += 1

    write_jsonl(args.out_manifest, out_rows)
    summary = {
        "manifest": str(args.manifest),
        "roi_manifest": str(args.roi_manifest),
        "old_roi_root": str(args.old_roi_root),
        "clean_root": str(args.clean_root),
        "out_root": str(args.out_root),
        "out_manifest": str(args.out_manifest),
        "stats": dict(stats),
        "errors_count": len(errors),
        "errors_sample": errors[:20],
    }
    summary_path = args.out_root / "recrop_v3_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"recrop finished with {len(errors)} errors; see {summary_path}")


if __name__ == "__main__":
    main()
