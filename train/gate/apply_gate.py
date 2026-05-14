"""Apply learned gate: load model, predict use_s2, output decisions + final metrics.

Inputs:
  gate_config.json           — feature policy, threshold, drop prefixes
  gate_model.joblib          — preprocessor + RF model
  gate_training_table.csv    — feature__ columns + target__ columns (for eval)

Outputs:
  outputs/final/gate_decisions.csv   — per-finding decisions
  outputs/final/final_metrics.json   — micro Dice + comparison baselines
"""

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn; sklearn.set_config(transform_output='default')

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "gate_training" / "gate_config.json"


def micro_dice(t, p, g): return 2.0 * t / max(1.0, p + g)


def main():
    if not CONFIG_PATH.exists():
        print(f"[ERROR] gate_config.json not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        gate_cfg = json.load(f)

    model_path = ROOT / gate_cfg["model_path"]
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Run: python gate_training/train_gate.py --fit-final")
        sys.exit(1)

    table_path = ROOT / gate_cfg["table_path"]
    if not table_path.exists():
        print(f"[ERROR] Table not found: {table_path}")
        print("  Run: python gate_training/build_gate_training_table.py")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    bundle = joblib.load(model_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    threshold = gate_cfg["threshold"]

    metadata_path = ROOT / gate_cfg["metadata_path"]
    if not metadata_path.exists():
        print(f"[ERROR] Metadata not found: {metadata_path}")
        print("  Run: python gate_training/train_gate.py --fit-final")
        sys.exit(1)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    kept = metadata["feature_cols"]

    # ── Load CSV ──
    cols, rows = [], []
    with open(table_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cols = list(r.keys())
            rows.append(r)
    missing = [c for c in kept if c not in cols]
    if missing:
        print(f"[ERROR] Table is missing {len(missing)} model features, first: {missing[:5]}")
        sys.exit(1)
    print(f"Table: {len(rows)} rows, {len(kept)} model features "
          f"(policy: {metadata.get('feature_policy', gate_cfg['feature_policy'])})")

    # ── Build X ──
    cat_in = [c for c in kept if c in [
        'feature__pred_category','feature__anatomy_target','feature__laterality',
        'feature__anatomy_group','feature__final_policy','feature__final_tightness','feature__expert']]
    num_in = [c for c in kept if c not in cat_in]
    nc = len(cat_in)
    n_num = len(num_in)
    X = np.empty((len(rows), nc + n_num), dtype=object)
    for i in range(nc):
        X[:, i] = [r.get(cat_in[i], "") for r in rows]
    for j in range(n_num):
        X[:, nc + j] = np.array([float(r.get(num_in[j], 0) or 0) for r in rows], dtype=np.float64)

    X_proc = preprocessor.transform(X)
    p_use_s2 = model.predict_proba(np.asarray(X_proc))[:, 1]
    use_s2 = (p_use_s2 >= threshold).astype(int)

    # ── Apply gate ──
    s1_tp = np.array([float(r.get("target__s1_tp", 0)) for r in rows], dtype=np.float64)
    s1_pred = np.array([float(r.get("target__s1_pred_voxels", 0)) for r in rows], dtype=np.float64)
    s2_tp = np.array([float(r.get("target__s2_raw_tp", 0)) for r in rows], dtype=np.float64)
    s2_pred = np.array([float(r.get("target__s2_raw_pred_voxels", 0)) for r in rows], dtype=np.float64)
    gt = np.array([float(r.get("target__gt_voxels", 0)) for r in rows], dtype=np.float64)

    tp_sum = np.where(use_s2, s2_tp, s1_tp).sum()
    pred_sum = np.where(use_s2, s2_pred, s1_pred).sum()
    gt_sum = gt.sum()
    gate_dice = micro_dice(tp_sum, pred_sum, gt_sum)
    s1_dice = micro_dice(float(s1_tp.sum()), float(s1_pred.sum()), float(gt_sum))
    s2_raw_dice = micro_dice(float(s2_tp.sum()), float(s2_pred.sum()), float(gt_sum))

    print(f"Gate micro Dice: {gate_dice:.4f}  (n_use_s2={use_s2.sum()}, threshold={threshold})")
    print(f"  Always S1:         {s1_dice:.4f}")
    print(f"  Always S2_raw:     {s2_raw_dice:.4f}")
    print(f"  Rule gate v2:      see eval_finding_level.json")

    # ── Write decisions ──
    decisions_out = ROOT / gate_cfg["decisions_out"]
    decisions_out.parent.mkdir(parents=True, exist_ok=True)
    with open(decisions_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["finding_id", "category", "p_use_s2", "use_s2",
                          "s1_dice", "s2_raw_dice", "s1_pred_voxels", "s2_raw_pred_voxels"])
        for i, r in enumerate(rows):
            writer.writerow([
                r["meta__finding_id"],
                r["meta__category"],
                round(float(p_use_s2[i]), 4),
                int(use_s2[i]),
                r.get("target__s1_dice", ""),
                r.get("target__s2_raw_dice", ""),
                int(float(r.get("target__s1_pred_voxels", 0))),
                int(float(r.get("target__s2_raw_pred_voxels", 0))),
            ])
    print(f"Decisions -> {decisions_out}")

    # ── Write metrics ──
    metrics_out = ROOT / gate_cfg["metrics_out"]
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    final_metrics = {
        "gate_micro_dice": round(gate_dice, 4),
        "always_s1_micro_dice": round(s1_dice, 4),
        "always_s2_raw_micro_dice": round(s2_raw_dice, 4),
        "n_use_s2": int(use_s2.sum()),
        "n_total": len(rows),
        "threshold": threshold,
        "feature_policy": gate_cfg["feature_policy"],
        "tp": int(tp_sum), "pred_voxels": int(pred_sum), "gt_voxels": int(gt_sum),
    }
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics -> {metrics_out}")


if __name__ == "__main__":
    main()
