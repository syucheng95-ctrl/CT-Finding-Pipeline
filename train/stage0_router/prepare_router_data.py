import argparse
import json
from collections import Counter
from pathlib import Path


CATEGORY_NAMES = {
    "1a": "bronchial_wall_thickening",
    "1b": "bronchiectasis",
    "1c": "emphysema",
    "1d": "interlobular_septal_thickening",
    "1e": "micronodule",
    "1f": "other_non_focal",
    "2a": "linear_scar",
    "2b": "atelectasis_consolidation",
    "2c": "ground_glass_opacity",
    "2d": "nodule_mass",
    "2e": "pleural_effusion",
    "2f": "honeycombing",
    "2g": "pneumothorax",
    "2h": "other_focal",
}

TIGHTNESS_MAP = {
    "1a": "conservative",
    "1b": "conservative",
    "1c": "conservative",
    "1d": "conservative",
    "1e": "conservative",
    "1f": "conservative",
    "2e": "conservative",
    "2f": "conservative",
    "2g": "conservative",
    "2a": "moderate",
    "2b": "moderate",
    "2c": "moderate",
    "2h": "moderate",
    "2d": "aggressive",
}


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_rows(rows, source_path):
    out = []
    for row in rows:
        category = str(row["category"])
        out.append(
            {
                "id": row["id"],
                "case_name": row.get("case_name"),
                "prompt": row["prompt"],
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "unknown"),
                "tightness": TIGHTNESS_MAP.get(category, "unknown"),
                "split": row.get("split"),
                "source_manifest": str(source_path),
            }
        )
    return out


def summarize(rows):
    return {
        "rows": len(rows),
        "category_counts": dict(sorted(Counter(r["category"] for r in rows).items())),
        "tightness_counts": dict(sorted(Counter(r["tightness"] for r in rows).items())),
        "missing_prompt": sum(1 for r in rows if not r.get("prompt")),
        "unknown_category": sum(1 for r in rows if r.get("category_name") == "unknown"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="../生医工大赛-demo/full_dataset_work/clean_full",
        help="Directory containing manifest_all_stage2split_*.jsonl.",
    )
    parser.add_argument("--out-dir", default="router_training/data")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    splits = {
        "train": data_root / "manifest_all_stage2split_train.jsonl",
        "val": data_root / "manifest_all_stage2split_val.jsonl",
        "holdout": data_root / "manifest_all_stage2split_holdout.jsonl",
    }

    summary = {
        "task": "Train a prompt Router to predict the 14 paper lesion categories; crop policy is derived later by rules.",
        "category_names": CATEGORY_NAMES,
        "tightness_map_v1": TIGHTNESS_MAP,
        "splits": {},
    }
    for split, path in splits.items():
        rows = normalize_rows(read_jsonl(path), path)
        write_jsonl(rows, out_dir / f"router_{split}.jsonl")
        summary["splits"][split] = summarize(rows)

    all_rows = []
    for split in ("train", "val", "holdout"):
        all_rows.extend(read_jsonl(out_dir / f"router_{split}.jsonl"))
    summary["all"] = summarize(all_rows)

    (out_dir / "label_map.json").write_text(
        json.dumps(CATEGORY_NAMES, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "policy_map_v1.json").write_text(
        json.dumps(TIGHTNESS_MAP, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
