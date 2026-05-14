import argparse
import json
import os
import shutil
import ssl
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def tqdm_disabled():
    return os.environ.get("V3_RECROP_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}


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


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def case_name_from_image_path(value: str):
    return Path(value).name


def finding_name_from_label_path(value: str):
    return Path(value).name


def hwd_from_zyx_box(box_zyx):
    z1, h1, w1, z2, h2, w2 = [int(v) for v in box_zyx]
    return [h1, w1, z1, h2, w2, z2]


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


def resolve_old_roi(old_roi_root: Path, rel: str):
    path = Path(rel)
    if path.is_absolute():
        return path
    return old_roi_root / path


def copy_existing_roi(row, old_roi_root: Path, out_root: Path):
    image_rel = row.get("roi_image_path") or row.get("roi_image")
    mask_rel = row.get("roi_mask_path") or row.get("roi_mask")
    img_src = resolve_old_roi(old_roi_root, image_rel)
    mask_src = resolve_old_roi(old_roi_root, mask_rel)
    img_dst = out_root / image_rel
    mask_dst = out_root / mask_rel
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    if not img_dst.exists() or img_dst.stat().st_size == 0:
        shutil.copy2(img_src, img_dst)
    if not mask_dst.exists() or mask_dst.stat().st_size == 0:
        shutil.copy2(mask_src, mask_dst)


def output_files_exist(row, out_root: Path):
    image_rel = row.get("roi_image_path") or row.get("roi_image")
    mask_rel = row.get("roi_mask_path") or row.get("roi_mask")
    if not image_rel or not mask_rel:
        return False
    image_path = out_root / image_rel
    mask_path = out_root / mask_rel
    return (
        image_path.exists()
        and mask_path.exists()
        and image_path.stat().st_size > 0
        and mask_path.stat().st_size > 0
    )


def make_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_modelscope_file(repo_id: str, file_path: str, out_path: Path, revision: str, retries: int, backoff: float):
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://www.modelscope.cn/api/v1/datasets/{repo_id}/repo"
        f"?Revision={urllib.parse.quote(revision)}&FilePath={urllib.parse.quote(file_path, safe='')}"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    token = os.environ.get("MODELSCOPE_TOKEN") or os.environ.get("MODELSCOPE_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error = None
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=make_ssl_context(), timeout=300) as resp:
                tmp.write_bytes(resp.read())
            os.replace(tmp, out_path)
            return out_path
        except Exception as exc:
            last_error = exc
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"failed to fetch {repo_id}:{file_path}: {last_error!r}")


def clean_image_remote_candidates(source_image: str):
    case_name = case_name_from_image_path(source_image)
    return [f"images/{case_name}", source_image]


def clean_label_remote_candidates(source_label: str):
    finding_name = finding_name_from_label_path(source_label)
    return [f"labels/{finding_name}", source_label]


def fetch_first_candidate(repo_id: str, candidates, cache_root: Path, revision: str, retries: int, backoff: float):
    errors = []
    for rel in candidates:
        out_path = cache_root / rel
        try:
            return fetch_modelscope_file(repo_id, rel, out_path, revision, retries, backoff)
        except Exception as exc:
            errors.append(f"{rel}: {exc!r}")
    raise RuntimeError("; ".join(errors))


def recrop_one(row, meta, image_nii, image, label_nii, label, out_root: Path, ratio: float, min_margin_hwd):
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
    new_row["source_image"] = meta.get("source_image") or row.get("source_image")
    new_row["source_label"] = meta.get("parent_label") or row.get("source_label")
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
    parser = argparse.ArgumentParser(description="Stream recrop large/xlarge ROIs without downloading clean-full.")
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest with train/val/holdout split.")
    parser.add_argument("--roi-manifest", type=Path, required=True, help="stage2-roi roi_manifest.jsonl with component boxes.")
    parser.add_argument("--old-roi-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, default=None)
    parser.add_argument("--clean-repo", default="clover259/ReXGroundingCT-clean-full")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--temp-root", type=Path, default=Path("cloud_work/_temp_clean_full"))
    parser.add_argument("--groups", nargs="+", default=["large", "xlarge"], choices=GROUP_ORDER)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--min-margin-hwd", nargs=3, type=int, default=[16, 16, 16])
    parser.add_argument("--copy-unchanged", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse rows already written to out-manifest and skip their files.")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--backoff", type=float, default=5.0)
    parser.add_argument("--progress-json", type=Path, default=None)
    parser.add_argument("--flush-every-cases", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.out_manifest is None:
        args.out_manifest = args.out_root / "manifest_v3.jsonl"
    if args.progress_json is None:
        args.progress_json = args.out_root / "recrop_v3_stream_progress.json"

    rows = load_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    roi_meta = {component_key(row): row for row in load_jsonl(args.roi_manifest)}
    completed = {}
    if args.resume and args.out_manifest.exists():
        for row in load_jsonl(args.out_manifest):
            if output_files_exist(row, args.out_root):
                completed[component_key(row)] = row

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "roi_images").mkdir(parents=True, exist_ok=True)
    (args.out_root / "roi_masks").mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)

    target_groups = set(args.groups)
    by_case = defaultdict(list)
    unchanged_rows = []
    out_rows_by_key = {}
    stats = Counter()
    errors = []

    for index, row in enumerate(rows):
        key = component_key(row)
        group = get_group(row["roi_shape_hwd"])
        if key in completed:
            out_rows_by_key[index] = completed[key]
            stats[f"resume_{group}"] += 1
            continue
        if group in target_groups:
            if key not in roi_meta:
                errors.append({"id": key, "group": group, "error": "missing roi metadata"})
                stats["error"] += 1
                continue
            meta = roi_meta[key]
            source_image = meta.get("source_image") or row.get("source_image")
            case_name = case_name_from_image_path(source_image)
            by_case[case_name].append((index, row, meta, group))
        else:
            new_row = dict(row)
            new_row["v3_recrop"] = False
            unchanged_rows.append((index, new_row))
            stats[f"keep_{group}"] += 1

    if args.copy_unchanged:
        for _index, row in tqdm(unchanged_rows, desc="copy unchanged ROI", disable=tqdm_disabled()):
            copy_existing_roi(row, args.old_roi_root, args.out_root)

    for index, row in unchanged_rows:
        out_rows_by_key[index] = row

    case_list = sorted(by_case.items())
    total_cases = len(case_list)
    processed_cases = 0

    def flush_progress(done: bool = False):
        partial_rows = [out_rows_by_key[i] for i in sorted(out_rows_by_key)]
        write_jsonl(args.out_manifest, partial_rows)
        write_json(
            args.progress_json,
            {
                "done": done,
                "processed_cases": processed_cases,
                "total_cases": total_cases,
                "processed_rows": len(partial_rows),
                "input_rows": len(rows),
                "stats": dict(stats),
                "errors_count": len(errors),
                "errors_sample": errors[:20],
                "out_manifest": str(args.out_manifest),
            },
        )

    for case_name, case_items in tqdm(case_list, desc="stream recrop cases", disable=tqdm_disabled()):
        image_source = case_items[0][2].get("source_image") or case_items[0][1].get("source_image")
        image_path = None
        try:
            image_path = fetch_first_candidate(
                args.clean_repo,
                clean_image_remote_candidates(image_source),
                args.temp_root,
                args.revision,
                args.retries,
                args.backoff,
            )
            image_nii = nib.load(str(image_path))
            image = np.asanyarray(image_nii.dataobj)
            if image.ndim == 4:
                image = image[..., 0]

            label_cache = {}
            for index, row, meta, group in case_items:
                label_source = meta.get("parent_label") or row.get("source_label")
                try:
                    if label_source not in label_cache:
                        label_path = fetch_first_candidate(
                            args.clean_repo,
                            clean_label_remote_candidates(label_source),
                            args.temp_root,
                            args.revision,
                            args.retries,
                            args.backoff,
                        )
                        label_nii = nib.load(str(label_path))
                        label = np.asanyarray(label_nii.dataobj)
                        if label.ndim == 4:
                            label = label[..., 0]
                        if image.shape != label.shape:
                            raise ValueError(f"image shape {image.shape} != label shape {label.shape}")
                        label_cache[label_source] = (label_nii, label, label_path)

                    label_nii, label, _label_path = label_cache[label_source]
                    new_row = recrop_one(row, meta, image_nii, image, label_nii, label, args.out_root, args.ratio, tuple(args.min_margin_hwd))
                    out_rows_by_key[index] = new_row
                    stats[f"recrop_{group}"] += 1
                except Exception as exc:
                    key = component_key(row)
                    errors.append({"id": key, "case_name": case_name, "group": group, "error": repr(exc)})
                    stats["error"] += 1

            if not args.keep_temp:
                try:
                    image_path.unlink()
                except FileNotFoundError:
                    pass
                for _label_nii, _label, label_path in label_cache.values():
                    try:
                        Path(label_path).unlink()
                    except FileNotFoundError:
                        pass
        except Exception as exc:
            for _index, row, _meta, group in case_items:
                errors.append({"id": component_key(row), "case_name": case_name, "group": group, "error": repr(exc)})
                stats["error"] += 1
        processed_cases += 1
        if processed_cases % max(1, args.flush_every_cases) == 0:
            flush_progress(done=False)

    out_rows = [out_rows_by_key[i] for i in sorted(out_rows_by_key)]
    write_jsonl(args.out_manifest, out_rows)
    summary = {
        "manifest": str(args.manifest),
        "roi_manifest": str(args.roi_manifest),
        "old_roi_root": str(args.old_roi_root),
        "clean_repo": args.clean_repo,
        "out_root": str(args.out_root),
        "out_manifest": str(args.out_manifest),
        "temp_root": str(args.temp_root),
        "keep_temp": args.keep_temp,
        "stats": dict(stats),
        "output_rows": len(out_rows),
        "input_rows": len(rows),
        "errors_count": len(errors),
        "errors_sample": errors[:20],
    }
    summary_path = args.out_root / "recrop_v3_stream_summary.json"
    write_json(summary_path, summary)
    flush_progress(done=(len(errors) == 0))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"recrop finished with {len(errors)} errors; see {summary_path}")


if __name__ == "__main__":
    main()
