"""Train a learned gate with GroupKFold CV.

Key design:
- White-list: only feature__ columns go into the model.
- Per-fold Dinkelbach lambda & dynamic label (no GT leakage).
- ColumnTransformer for one-hot categoricals + numeric scaling.
- Threshold scan on train fold, applied to test fold.
- Compares: Always S1, Rule gate v2, LR, RF, Oracle raw.

No-pandas version (compatible with numpy 2.x).
"""

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# Disable pandas-based output to avoid numpy 1.x/2.x conflicts with pandas C extensions
sklearn.set_config(transform_output="default")

warnings.filterwarnings("ignore")

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent  # upload/ directory
CSV_PATH = ROOT / "gate" / "outputs" / "gate_training_table.csv"
OUT_JSON = ROOT / "gate" / "outputs" / "gate_cv_results.json"

# Categorical columns (stored as strings in CSV, one-hot in pipeline)
CATEGORICAL_COLS = [
    "feature__pred_category",
    "feature__anatomy_target",
    "feature__laterality",
    "feature__anatomy_group",
    "feature__final_policy",
    "feature__final_tightness",
    "feature__expert",
]

# Rule gate v2 constants
MAX_OVERSEGMENTATION_RATIO = 5.0
MIN_COARSE_VOXELS_FOR_GATE = 100


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Load CSV into column names and list of row dicts (all values as strings)."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    return cols, rows


def micro_dice(tp, pred, gt):
    return (2.0 * tp) / max(1.0, pred + gt)


def dinkelbach(s1_tp, s1_pred, s2_tp, s2_pred, gt, max_iter=50, tol=1e-8):
    """Dinkelbach iteration: find oracle lambda & labels for a set of findings."""
    s1_tp = np.asarray(s1_tp, dtype=np.float64)
    s1_pred = np.asarray(s1_pred, dtype=np.float64)
    s2_tp = np.asarray(s2_tp, dtype=np.float64)
    s2_pred = np.asarray(s2_pred, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)

    tp_sum = s1_tp.sum()
    pred_sum = s1_pred.sum()
    gt_sum = gt_arr.sum()
    lam = (2.0 * tp_sum) / max(1.0, pred_sum + gt_sum)

    delta_tp = s2_tp - s1_tp
    delta_pred = s2_pred - s1_pred

    for _ in range(max_iter):
        gain = 2.0 * delta_tp - lam * delta_pred
        use_s2 = gain > 0
        tp = np.where(use_s2, s2_tp, s1_tp).sum()
        pred = np.where(use_s2, s2_pred, s1_pred).sum()
        lam_new = (2.0 * tp) / max(1.0, pred + gt_sum)
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new

    labels = (2.0 * delta_tp - lam * delta_pred) > 0
    return lam, labels.astype(int)


def apply_gate(use_s2, s1_tp, s1_pred, s2_tp, s2_pred, gt):
    """Apply gate decisions and return micro-aggregated counts."""
    use_s2 = np.asarray(use_s2, dtype=bool)
    s1_tp = np.asarray(s1_tp, dtype=np.float64)
    s1_pred = np.asarray(s1_pred, dtype=np.float64)
    s2_tp = np.asarray(s2_tp, dtype=np.float64)
    s2_pred = np.asarray(s2_pred, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)

    tp = np.where(use_s2, s2_tp, s1_tp).sum()
    pred = np.where(use_s2, s2_pred, s1_pred).sum()
    gt_sum = gt_arr.sum()
    return tp, pred, gt_sum


def rule_gate_v2(s1_pred_vox, s2_pred_vox):
    """Apply hand-written rule gate v2. Returns use_s2 boolean array."""
    s1_pred = np.asarray(s1_pred_vox, dtype=np.float64)
    s2_pred = np.asarray(s2_pred_vox, dtype=np.float64)
    gated = (s1_pred >= MIN_COARSE_VOXELS_FOR_GATE) & (
        (s2_pred == 0) |
        (s2_pred > MAX_OVERSEGMENTATION_RATIO * s1_pred) |
        (s2_pred < 0.2 * s1_pred)
    )
    return ~gated


def classification_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2.0 * prec * rec) / max(1e-9, prec + rec)
    bal_acc = 0.5 * ((tp / max(1, tp + fn)) + (tn / max(1, tn + fp)))
    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "n_pred_use_s2": int((y_pred == 1).sum()),
    }


def threshold_scan(probs, s1_tp, s1_pred, s2_tp, s2_pred, gt):
    """Scan thresholds on train fold, return (best_threshold, best_micro_dice)."""
    best_thresh = 0.5
    best_dice = 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        use_s2 = probs >= t
        if use_s2.sum() == 0:
            dice = micro_dice(float(s1_tp.sum()), float(s1_pred.sum()), float(gt.sum()))
        elif use_s2.sum() == len(use_s2):
            dice = micro_dice(float(s2_tp.sum()), float(s2_pred.sum()), float(gt.sum()))
        else:
            tp_sum, pred_sum, gt_sum = apply_gate(use_s2, s1_tp, s1_pred, s2_tp, s2_pred, gt)
            dice = micro_dice(tp_sum, pred_sum, gt_sum)
        if dice > best_dice:
            best_dice = dice
            best_thresh = float(t)
    return best_thresh, best_dice


def fit_and_evaluate(model, X_train, y_train, X_test,
                     t_train_s1_tp, t_train_s1_pred, t_train_s2_tp, t_train_s2_pred, t_train_gt,
                     e_s1_tp, e_s1_pred, e_s2_tp, e_s2_pred, e_gt,
                     label_test_report, feat_names=None):
    """Train one model, threshold-scan on train, eval on test."""
    model.fit(X_train, y_train)
    probs_train = model.predict_proba(X_train)[:, 1]
    probs_test = model.predict_proba(X_test)[:, 1]

    best_thresh, train_best_dice = threshold_scan(
        probs_train, t_train_s1_tp, t_train_s1_pred,
        t_train_s2_tp, t_train_s2_pred, t_train_gt)

    y_pred = (probs_test >= best_thresh).astype(int)
    tp_sum, pred_sum, gt_sum = apply_gate(
        y_pred, e_s1_tp, e_s1_pred, e_s2_tp, e_s2_pred, e_gt)
    test_dice = micro_dice(tp_sum, pred_sum, gt_sum)

    cls_metrics = classification_metrics(label_test_report, y_pred)

    result = {
        "best_threshold": round(best_thresh, 2),
        "train_micro_dice_at_best": round(train_best_dice, 4),
        "test_micro_dice": round(test_dice, 4),
        "test_tp": int(tp_sum), "test_pred": int(pred_sum), "test_gt": int(gt_sum),
    }
    result.update(cls_metrics)

    if hasattr(model, "feature_importances_") and feat_names is not None:
        importance = sorted(
            zip(feat_names, model.feature_importances_),
            key=lambda x: -x[1])[:20]
        result["feature_importance"] = {str(k): round(float(v), 6) for k, v in importance}

    return result


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data (no pandas) ──
    cols, row_dicts = load_csv(CSV_PATH)
    print(f"Loaded {len(row_dicts)} rows, {len(cols)} columns")

    feature_cols = [c for c in cols if c.startswith("feature__")]
    # Drop ROI features (noisy per ablation) and overlap features (didn't help)
    drop_prefixes = ("feature__roi_", "feature__n_rois", "feature__n_coarse_rois",
                     "feature__n_proposal_rois", "feature__roi_coarse_fraction",
                     "feature__s1_s2_", "feature__s2_overlap", "feature__s1_covered",
                     "feature__s2_outside", "feature__s1_outside")
    feature_cols = [c for c in feature_cols if not c.startswith(drop_prefixes)]
    print(f"  feature__ cols: {len(feature_cols)}")

    # Build arrays
    groups = np.array([r["meta__case_name"] for r in row_dicts])
    s1_tp_all = np.array([float(r.get("target__s1_tp", 0)) for r in row_dicts], dtype=np.float64)
    s1_pred_all = np.array([float(r.get("target__s1_pred_voxels", 0)) for r in row_dicts], dtype=np.float64)
    s2_tp_all = np.array([float(r.get("target__s2_raw_tp", 0)) for r in row_dicts], dtype=np.float64)
    s2_pred_all = np.array([float(r.get("target__s2_raw_pred_voxels", 0)) for r in row_dicts], dtype=np.float64)
    gt_all = np.array([float(r.get("target__gt_voxels", 0)) for r in row_dicts], dtype=np.float64)

    # Build feature matrix as list-of-lists (strings where categorical, float where numeric)
    # We store as list of dicts grouped by type
    cat_cols_present = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols_present]

    # Build X as structured arrays: categorical parts as list-of-strings, numeric as float
    X_cat_data = [[r.get(c, "") for c in cat_cols_present] for r in row_dicts]
    X_num_data = [[float(r.get(c, 0) or 0) for c in num_cols] for r in row_dicts]

    X_cat_arr = np.array(X_cat_data, dtype=object)
    X_num_arr = np.array(X_num_data, dtype=np.float64)

    # Stack for GroupKFold indexing (KFold just needs an array with shape[0])
    # We'll use the numeric array for indexing, but process categorical separately
    X_for_split = X_num_arr  # just for getting indices

    print(f"  categorical: {len(cat_cols_present)} ({cat_cols_present})")
    print(f"  numeric:     {len(num_cols)}")

    # ── Baselines ──
    always_s1_dice = micro_dice(
        float(s1_tp_all.sum()), float(s1_pred_all.sum()), float(gt_all.sum()))
    print(f"\nAlways S1 micro Dice: {always_s1_dice:.4f}")

    use_s2_rule = rule_gate_v2(s1_pred_all, s2_pred_all)
    tp_r, pred_r, gt_r = apply_gate(use_s2_rule, s1_tp_all, s1_pred_all, s2_tp_all, s2_pred_all, gt_all)
    rule_dice = micro_dice(tp_r, pred_r, gt_r)
    print(f"Rule gate v2 micro Dice: {rule_dice:.4f}  (n_use_s2={use_s2_rule.sum()})")

    lam_global, label_oracle = dinkelbach(s1_tp_all, s1_pred_all, s2_tp_all, s2_pred_all, gt_all)
    tp_o, pred_o, gt_o = apply_gate(label_oracle, s1_tp_all, s1_pred_all, s2_tp_all, s2_pred_all, gt_all)
    oracle_dice = micro_dice(tp_o, pred_o, gt_o)
    print(f"Oracle raw micro Dice: {oracle_dice:.4f}  (lambda={lam_global:.6f}, n_use_s2={label_oracle.sum()})")

    # ── Models ──
    scale_weight = (s1_tp_all.shape[0] - 42) / max(1, 42)  # ~5.6
    model_factories = {
        "RF_depth4": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1),
    }

    # ── Build preprocessor ──
    n_cat = len(cat_cols_present)
    n_num = len(num_cols)
    cat_indices = list(range(n_cat))
    num_indices = list(range(n_cat, n_cat + n_num))

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_indices),
        ("num", StandardScaler(), num_indices),
    ])

    # Combine cat + num for ColumnTransformer input
    # ColumnTransformer expects a 2D array-like where each row is one sample
    # We need to interleave cat and num columns correctly
    # Actually, ColumnTransformer takes a dataframe-like input. We'll construct
    # a 2D object array where first len(cat_cols) cols are strings, rest are floats.
    n_samples = len(row_dicts)
    n_cat = len(cat_cols_present)
    n_num = len(num_cols)
    X_combined = np.empty((n_samples, n_cat + n_num), dtype=object)
    for i in range(n_cat):
        X_combined[:, i] = X_cat_arr[:, i]
    for j in range(n_num):
        X_combined[:, n_cat + j] = X_num_arr[:, j]

    # ── GroupKFold ──
    gkf = GroupKFold(n_splits=5)
    results = {
        "always_s1_micro_dice": round(always_s1_dice, 4),
        "rule_gate_v2_micro_dice": round(rule_dice, 4),
        "rule_gate_v2_n_use_s2": int(use_s2_rule.sum()),
        "oracle_raw_micro_dice": round(oracle_dice, 4),
        "oracle_n_use_s2": int(label_oracle.sum()),
        "oracle_lambda": round(lam_global, 6),
        "models": {},
    }

    all_fold_results = {name: [] for name in model_factories}

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_for_split, groups=groups)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_i + 1}: train={len(train_idx)}, test={len(test_idx)}")

        X_train_raw = X_combined[train_idx]
        X_test_raw = X_combined[test_idx]

        t_s1_tp = s1_tp_all[train_idx];   e_s1_tp = s1_tp_all[test_idx]
        t_s1_pred = s1_pred_all[train_idx]; e_s1_pred = s1_pred_all[test_idx]
        t_s2_tp = s2_tp_all[train_idx];    e_s2_tp = s2_tp_all[test_idx]
        t_s2_pred = s2_pred_all[train_idx]; e_s2_pred = s2_pred_all[test_idx]
        t_gt = gt_all[train_idx];          e_gt = gt_all[test_idx]

        # ── Per-fold Dinkelbach on train ──
        lam_fold, label_train = dinkelbach(t_s1_tp, t_s1_pred, t_s2_tp, t_s2_pred, t_gt)
        print(f"  lambda_train = {lam_fold:.6f}  n_use_s2 = {label_train.sum()}/{len(label_train)}")

        # ── label_test_for_report (only for classification metrics) ──
        _, label_test_report = dinkelbach(e_s1_tp, e_s1_pred, e_s2_tp, e_s2_pred, e_gt)

        # ── Preprocessing ──
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        # Get column names after one-hot
        cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols_present)
        all_feat_names = list(cat_names) + num_cols

        # ── Train & evaluate each model ──
        for name, make_model in model_factories.items():
            model = make_model()
            X_train_arr = np.asarray(X_train)
            X_test_arr = np.asarray(X_test)

            res = fit_and_evaluate(
                model, X_train_arr, label_train, X_test_arr,
                t_s1_tp, t_s1_pred, t_s2_tp, t_s2_pred, t_gt,
                e_s1_tp, e_s1_pred, e_s2_tp, e_s2_pred, e_gt,
                label_test_report, feat_names=all_feat_names)

            res["fold"] = fold_i + 1
            res["lambda_fold"] = round(lam_fold, 6)
            res["n_train"] = int(len(train_idx))
            res["n_test"] = int(len(test_idx))
            res["n_train_use_s2"] = int(label_train.sum())

            all_fold_results[name].append(res)
            print(f"  {name}: test_dice={res['test_micro_dice']:.4f}  "
                  f"thresh={res['best_threshold']:.2f}  "
                  f"n_pred_s2={res['n_pred_use_s2']}  "
                  f"prec={res['precision']:.3f}  rec={res['recall']:.3f}")

    # ── Aggregate ──
    for name in model_factories:
        folds = all_fold_results[name]
        dice_vals = [f["test_micro_dice"] for f in folds]
        # Pooled OOF: sum all tp/pred/gt across folds, compute ONE micro Dice
        pooled_tp = sum(f["test_tp"] for f in folds)
        pooled_pred = sum(f["test_pred"] for f in folds)
        pooled_gt = sum(f["test_gt"] for f in folds)
        pooled_oof_dice = (2 * pooled_tp) / max(1, pooled_pred + pooled_gt)

        results["models"][name] = {
            "fold_micro_dice": [round(float(v), 4) for v in dice_vals],
            "mean_micro_dice": round(float(np.mean(dice_vals)), 4),
            "std_micro_dice": round(float(np.std(dice_vals)), 4),
            "pooled_oof_micro_dice": round(pooled_oof_dice, 4),
            "pooled_oof_tp": pooled_tp, "pooled_oof_pred": pooled_pred, "pooled_oof_gt": pooled_gt,
            "use_s2_precision": round(float(np.mean([f["precision"] for f in folds])), 4),
            "use_s2_recall": round(float(np.mean([f["recall"] for f in folds])), 4),
            "use_s2_f1": round(float(np.mean([f["f1"] for f in folds])), 4),
            "n_pred_use_s2_mean": float(np.mean([f["n_pred_use_s2"] for f in folds])),
            "best_thresholds": [f["best_threshold"] for f in folds],
            "folds": folds,
        }
        if "feature_importance" in folds[0]:
            results["models"][name]["feature_importance_last_fold"] = folds[0]["feature_importance"]

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults -> {OUT_JSON}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Gate':<25} {'fold_mean':>12} {'pooled_OOF':>12} {'n_use_s2':>10}")
    print("-" * 62)
    print(f"{'Always S1':<25} {'-':>12} {results['always_s1_micro_dice']:>12.4f} {'-':>10}")
    print(f"{'Rule gate v2':<25} {'-':>12} {results['rule_gate_v2_micro_dice']:>12.4f} {results['rule_gate_v2_n_use_s2']:>10}")
    print(f"{'Oracle raw':<25} {'-':>12} {results['oracle_raw_micro_dice']:>12.4f} {results['oracle_n_use_s2']:>10}")
    for name in model_factories:
        m = results["models"][name]
        print(f"{name:<25} {m['mean_micro_dice']:>12.4f} {m['pooled_oof_micro_dice']:>12.4f} {m['n_pred_use_s2_mean']:>8.0f}")


import argparse, joblib


def fit_final():
    """Train final gate model on ALL data, save model + metadata."""
    config_path = ROOT / "gate" / "gate_config.json"
    if not config_path.exists():
        print(f"[ERROR] gate_config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        gate_cfg = json.load(f)

    drop_prefixes = tuple(gate_cfg["drop_prefixes"])
    threshold = gate_cfg["threshold"]
    model_params = gate_cfg["model_params"]

    cols, row_dicts = load_csv(CSV_PATH)
    feature_cols = [c for c in cols if c.startswith("feature__")]
    feature_cols = [c for c in feature_cols if not c.startswith(drop_prefixes)]
    print(f"Final model: {len(feature_cols)} features (policy: {gate_cfg['feature_policy']})")

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols_present]
    n_cat = len(cat_cols_present)
    n_num = len(num_cols)
    n_samples = len(row_dicts)

    X = np.empty((n_samples, n_cat + n_num), dtype=object)
    for i in range(n_cat):
        X[:, i] = [r.get(cat_cols_present[i], "") for r in row_dicts]
    for j in range(n_num):
        X[:, n_cat + j] = np.array([float(r.get(num_cols[j], 0) or 0) for r in row_dicts], dtype=np.float64)

    s1_tp_all = np.array([float(r.get("target__s1_tp", 0)) for r in row_dicts], dtype=np.float64)
    s1_pred_all = np.array([float(r.get("target__s1_pred_voxels", 0)) for r in row_dicts], dtype=np.float64)
    s2_tp_all = np.array([float(r.get("target__s2_raw_tp", 0)) for r in row_dicts], dtype=np.float64)
    s2_pred_all = np.array([float(r.get("target__s2_raw_pred_voxels", 0)) for r in row_dicts], dtype=np.float64)
    gt_all = np.array([float(r.get("target__gt_voxels", 0)) for r in row_dicts], dtype=np.float64)

    _, lab_all = dinkelbach(s1_tp_all, s1_pred_all, s2_tp_all, s2_pred_all, gt_all)
    print(f"Label: {lab_all.sum()}/{len(lab_all)} use_s2=1")

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(range(n_cat))),
        ("num", StandardScaler(), list(range(n_cat, n_cat + n_num))),
    ])
    X_proc = preprocessor.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=model_params.get("n_estimators", 100),
        max_depth=model_params.get("max_depth", 4),
        min_samples_leaf=model_params.get("min_samples_leaf", 5),
        class_weight=model_params.get("class_weight", "balanced"),
        random_state=42, n_jobs=-1,
    )
    model.fit(np.asarray(X_proc), lab_all)

    # Save model pipeline
    model_path = ROOT / gate_cfg["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "model": model}, model_path)
    print(f"Model saved -> {model_path}")

    # Save metadata
    metadata = {
        "feature_policy": gate_cfg["feature_policy"],
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols_present,
        "numeric_cols": num_cols,
        "drop_prefixes": list(drop_prefixes),
        "threshold": threshold,
        "model_type": gate_cfg["model_type"],
        "model_params": model_params,
        "n_samples": n_samples,
        "n_use_s2_label": int(lab_all.sum()),
    }
    metadata_path = ROOT / gate_cfg["metadata_path"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved -> {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-final", action="store_true", help="Train final model on all data and save")
    args, _ = parser.parse_known_args()
    if args.fit_final:
        fit_final()
    else:
        main()
