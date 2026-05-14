"""Ablation study: drop each feature group, measure impact on RF_depth4 micro Dice."""

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sklearn.set_config(transform_output="default")

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "gate_training" / "outputs" / "gate_training_table.csv"
OUT_JSON = ROOT / "gate_training" / "outputs" / "gate_ablation_results.json"

CATEGORICAL_COLS = [
    "feature__pred_category", "feature__anatomy_target", "feature__laterality",
    "feature__anatomy_group", "feature__final_policy", "feature__final_tightness",
    "feature__expert",
]

# Feature groups
FEATURE_GROUPS = {
    "A_keywords": lambda c: c.startswith("feature__kw_") or c.startswith("feature__prompt_"),
    "B_router": lambda c: c in [
        "feature__pred_category", "feature__category_confidence",
        "feature__anatomy_target", "feature__laterality", "feature__anatomy_group",
        "feature__final_policy", "feature__final_tightness", "feature__tightness_confidence",
        "feature__fail_open_reason_is_null", "feature__anatomy_fallback_reason_is_null",
        "feature__n_anatomy_lobes", "feature__is_both_lungs", "feature__is_conservative_crop",
    ],
    "C_crop": lambda c: c.startswith("feature__crop_"),
    "D_proposal": lambda c: c.startswith("feature__proposal_") or c in [
        "feature__expert", "feature__s05_fallback", "feature__n_proposals",
    ],
    "E_verifier": lambda c: c.startswith("feature__verified_") or c.startswith("feature__rejected_") or c in [
        "feature__n_verified", "feature__n_rejected", "feature__verified_ratio",
        "feature__s1_fallback", "feature__coarse_mask_count", "feature__s1_is_fullct_voxtell",
        "feature__verified_rejected_score_gap",
    ],
    "F_s1_coarse": lambda c: c.startswith("feature__s1_pred_"),
    "G_roi": lambda c: c.startswith("feature__roi_") or c in [
        "feature__n_rois", "feature__n_coarse_rois", "feature__n_proposal_rois",
        "feature__roi_coarse_fraction",
    ],
    "H_s2_raw": lambda c: c.startswith("feature__s2_"),
}


def micro_dice(tp, pred, gt):
    return (2.0 * tp) / max(1.0, pred + gt)


def dinkelbach(s1_tp, s1_pred, s2_tp, s2_pred, gt, max_iter=50, tol=1e-8):
    s1_tp = np.asarray(s1_tp, dtype=np.float64)
    s1_pred = np.asarray(s1_pred, dtype=np.float64)
    s2_tp = np.asarray(s2_tp, dtype=np.float64)
    s2_pred = np.asarray(s2_pred, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)
    lam = (2.0 * s1_tp.sum()) / max(1.0, s1_pred.sum() + gt_arr.sum())
    delta_tp = s2_tp - s1_tp
    delta_pred = s2_pred - s1_pred
    for _ in range(max_iter):
        gain = 2.0 * delta_tp - lam * delta_pred
        use_s2 = gain > 0
        tp = np.where(use_s2, s2_tp, s1_tp).sum()
        pred = np.where(use_s2, s2_pred, s1_pred).sum()
        lam_new = (2.0 * tp) / max(1.0, pred + gt_arr.sum())
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new
    return lam, ((2.0 * delta_tp - lam * delta_pred) > 0).astype(int)


def apply_gate(use_s2, s1_tp, s1_pred, s2_tp, s2_pred, gt):
    use_s2 = np.asarray(use_s2, dtype=bool)
    s1_tp = np.asarray(s1_tp, dtype=np.float64)
    s1_pred = np.asarray(s1_pred, dtype=np.float64)
    s2_tp = np.asarray(s2_tp, dtype=np.float64)
    s2_pred = np.asarray(s2_pred, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)
    tp = np.where(use_s2, s2_tp, s1_tp).sum()
    pred = np.where(use_s2, s2_pred, s1_pred).sum()
    return tp, pred, gt_arr.sum()


def threshold_scan(probs, s1_tp, s1_pred, s2_tp, s2_pred, gt):
    best_thresh, best_dice = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        use_s2 = probs >= t
        if use_s2.sum() == 0:
            dice = micro_dice(float(s1_tp.sum()), float(s1_pred.sum()), float(gt.sum()))
        elif use_s2.sum() == len(use_s2):
            dice = micro_dice(float(s2_tp.sum()), float(s2_pred.sum()), float(gt.sum()))
        else:
            tp, pred, gt_sum = apply_gate(use_s2, s1_tp, s1_pred, s2_tp, s2_pred, gt)
            dice = micro_dice(tp, pred, gt_sum)
        if dice > best_dice:
            best_dice = dice
            best_thresh = float(t)
    return best_thresh, best_dice


def evaluate_fold(X_train, X_test, y_train, r_train, r_test, fold_name):
    """Train RF_depth4 on one fold and return test micro dice."""
    model = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # Threshold scan on TRAIN
    t_s1_tp, t_s1_pred = r_train["s1_tp"], r_train["s1_pred"]
    t_s2_tp, t_s2_pred = r_train["s2_tp"], r_train["s2_pred"]
    t_gt = r_train["gt"]
    probs_train = model.predict_proba(X_train)[:, 1]
    best_thresh, _ = threshold_scan(probs_train, t_s1_tp, t_s1_pred, t_s2_tp, t_s2_pred, t_gt)

    # Apply on TEST
    use_s2 = probs >= best_thresh
    e_s1_tp, e_s1_pred = r_test["s1_tp"], r_test["s1_pred"]
    e_s2_tp, e_s2_pred = r_test["s2_tp"], r_test["s2_pred"]
    e_gt = r_test["gt"]
    tp, pred, gt_sum = apply_gate(use_s2, e_s1_tp, e_s1_pred, e_s2_tp, e_s2_pred, e_gt)
    return micro_dice(tp, pred, gt_sum), int(use_s2.sum())


def run_cv(feature_cols, df, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt):
    """Run 5-fold GroupKFold with given feature_cols, return list of test micro dices."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    n_cat = len(cat_cols)

    X_combined = np.empty((len(df), n_cat + len(num_cols)), dtype=object)
    for i in range(n_cat):
        X_combined[:, i] = [str(r.get(cat_cols[i], "")) for r in df]
    for j in range(len(num_cols)):
        X_combined[:, n_cat + j] = np.array([float(r.get(num_cols[j], 0) or 0) for r in df], dtype=np.float64)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(range(n_cat))),
        ("num", StandardScaler(), list(range(n_cat, n_cat + len(num_cols)))),
    ])

    gkf = GroupKFold(n_splits=5)
    dice_vals = []
    n_s2_vals = []

    for train_idx, test_idx in gkf.split(X_combined, groups=groups):
        X_train_raw = X_combined[train_idx]
        X_test_raw = X_combined[test_idx]

        r_train = {
            "s1_tp": s1_tp[train_idx], "s1_pred": s1_pred[train_idx],
            "s2_tp": s2_tp[train_idx], "s2_pred": s2_pred[train_idx], "gt": gt[train_idx],
        }
        r_test = {
            "s1_tp": s1_tp[test_idx], "s1_pred": s1_pred[test_idx],
            "s2_tp": s2_tp[test_idx], "s2_pred": s2_pred[test_idx], "gt": gt[test_idx],
        }

        lam_fold, label_train = dinkelbach(
            r_train["s1_tp"], r_train["s1_pred"], r_train["s2_tp"], r_train["s2_pred"], r_train["gt"])

        X_train_arr = np.asarray(preprocessor.fit_transform(X_train_raw))
        X_test_arr = np.asarray(preprocessor.transform(X_test_raw))

        dice, n_s2 = evaluate_fold(X_train_arr, X_test_arr, label_train, r_train, r_test, "")
        dice_vals.append(dice)
        n_s2_vals.append(n_s2)

    return dice_vals, n_s2_vals


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Load
    cols, all_rows = [], []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cols = list(r.keys())
            all_rows.append(r)
    print(f"Loaded {len(all_rows)} rows")

    all_feature_cols = [c for c in cols if c.startswith("feature__")]
    groups = np.array([r["meta__case_name"] for r in all_rows])
    s1_tp_all = np.array([float(r.get("target__s1_tp", 0)) for r in all_rows], dtype=np.float64)
    s1_pred_all = np.array([float(r.get("target__s1_pred_voxels", 0)) for r in all_rows], dtype=np.float64)
    s2_tp_all = np.array([float(r.get("target__s2_raw_tp", 0)) for r in all_rows], dtype=np.float64)
    s2_pred_all = np.array([float(r.get("target__s2_raw_pred_voxels", 0)) for r in all_rows], dtype=np.float64)
    gt_all = np.array([float(r.get("target__gt_voxels", 0)) for r in all_rows], dtype=np.float64)

    always_s1 = micro_dice(float(s1_tp_all.sum()), float(s1_pred_all.sum()), float(gt_all.sum()))

    print(f"Always S1: {always_s1:.4f}")
    print(f"{'='*80}")

    results = {"always_s1_micro_dice": round(always_s1, 4), "ablation": {}}

    # Full model
    print("Full model (all features)...")
    full_dice, full_n = run_cv(all_feature_cols, all_rows, groups, s1_tp_all, s1_pred_all, s2_tp_all, s2_pred_all, gt_all)
    full_mean = float(np.mean(full_dice))
    print(f"  {full_mean:.4f} +- {np.std(full_dice):.4f}  n_s2={float(np.mean(full_n)):.0f}\n")

    results["full_model"] = {
        "mean_dice": round(full_mean, 4),
        "std_dice": round(float(np.std(full_dice)), 4),
        "dice_per_fold": [round(v, 4) for v in full_dice],
        "n_features": len(all_feature_cols),
    }

    # Ablation
    print(f"{'Group':<20s} {'Features':>8s} {'Dice':>8s} {'Delta':>8s} {'n_s2':>6s}")
    print("-" * 52)

    for group_name, match_fn in FEATURE_GROUPS.items():
        dropped = [c for c in all_feature_cols if match_fn(c)]
        kept = [c for c in all_feature_cols if not match_fn(c)]
        print(f"{group_name:<20s} -{len(dropped):>3d} ...", end=" ", flush=True)

        dice_vals, n_s2_vals = run_cv(kept, all_rows, groups, s1_tp_all, s1_pred_all, s2_tp_all, s2_pred_all, gt_all)
        mean_dice = float(np.mean(dice_vals))
        delta = mean_dice - full_mean
        print(f"{mean_dice:>8.4f} {delta:>+8.4f} {float(np.mean(n_s2_vals)):>6.0f}")

        results["ablation"][group_name] = {
            "dropped": dropped,
            "n_kept": len(kept),
            "mean_dice": round(mean_dice, 4),
            "std_dice": round(float(np.std(dice_vals)), 4),
            "delta_vs_full": round(float(delta), 4),
            "dice_per_fold": [round(v, 4) for v in dice_vals],
        }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults -> {OUT_JSON}")


if __name__ == "__main__":
    main()
