"""Fine-grained ablation v2.

Two baselines:
  A: all 111 features
  B: current best = no_ROI_no_overlap (93 features)

Two experiments:
  1. Delete-one-subgroup from ALL features (which groups hurt?)
  2. Add-one-subgroup to BEST features (any ROI/overlap worth keeping?)
"""

import csv, json, warnings, numpy as np
from pathlib import Path
import sklearn; sklearn.set_config(transform_output='default')
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "gate_training" / "outputs" / "gate_training_table.csv"
OUT_JSON = ROOT / "gate_training" / "outputs" / "gate_ablation_v2_results.json"

CAT_COLS = ['feature__pred_category','feature__anatomy_target','feature__laterality',
            'feature__anatomy_group','feature__final_policy','feature__final_tightness','feature__expert']

# All subgroups with their feature names (without feature__ prefix)
ALL_SUBGROUPS = {
    'A1': ['kw_bilateral','kw_unilateral','kw_lobe_specified','kw_subpleural'],
    'A2': ['kw_diffuse','kw_focal','kw_multiple','kw_small','kw_mass'],
    'A3': ['kw_ggo','kw_nodule','kw_consolidation','kw_atelectasis','kw_bronchial','kw_pleural','kw_cavity','kw_tree_in_bud','kw_honeycomb'],
    'A4': ['kw_size_mentioned','prompt_len_chars','prompt_len_words'],
    'B1': ['pred_category','category_confidence'],
    'B2': ['anatomy_target','anatomy_group','laterality','n_anatomy_lobes','is_both_lungs'],
    'B3': ['final_policy','final_tightness','tightness_confidence','is_conservative_crop'],
    'B4': ['fail_open_reason_is_null','anatomy_fallback_reason_is_null'],
    'C1': ['crop_volume','crop_volume_ratio','crop_h','crop_w','crop_d','crop_is_fullish'],
    'C2': ['crop_center_h_norm','crop_center_w_norm','crop_center_d_norm'],
    'C3': ['crop_touches_h0','crop_touches_h1','crop_touches_w0','crop_touches_w1','crop_touches_d0','crop_touches_d1'],
    'C4': ['n_findings_in_crop_group'],
    'D1': ['expert'],
    'D2': ['n_proposals','s05_fallback'],
    'D3': ['proposal_max_volume','proposal_mean_volume','proposal_median_volume','proposal_total_volume','proposal_volume_std'],
    'D4': ['proposal_max_volume_ratio_to_crop','proposal_total_volume_ratio_to_crop'],
    'D5': ['proposal_source_hu_count','proposal_source_nodule_count','proposal_detector_score_max','proposal_detector_score_mean','proposal_hu_contrast_max'],
    'E1': ['n_verified','n_rejected','verified_ratio','s1_fallback'],
    'E2': ['verified_max_prob_max','verified_max_prob_mean','verified_mean_prob_mean','verified_score_max','verified_score_mean'],
    'E3': ['verified_fg_ratio_max','verified_fg_ratio_mean'],
    'E4': ['rejected_score_max','verified_rejected_score_gap','coarse_mask_count','s1_is_fullct_voxtell'],
    'F1': ['s1_pred_voxels','s1_pred_log_voxels','s1_pred_is_empty','s1_pred_is_huge'],
    'F2': ['s1_pred_volume_ratio_to_crop'],
    'G1': ['n_rois','n_coarse_rois','n_proposal_rois','roi_coarse_fraction'],
    'G2': ['roi_max_volume','roi_mean_volume','roi_median_volume','roi_total_volume','roi_volume_std'],
    'G3': ['roi_max_volume_ratio_to_crop','roi_total_volume_ratio_to_crop'],
    'G4': ['roi_shape_d_mean'],
    'H1': ['s2_raw_pred_voxels','s2_raw_log_pred_voxels','s2_pred_is_empty','s2_pred_is_huge'],
    'H2': ['s2_s1_volume_ratio','s2_roi_pred_density'],
    'H3': ['s2_overseg_ratio_gt5','s2_underseg_ratio_lt02'],
    'I1': ['s2_overlap_with_s1','s1_covered_by_s2','s1_s2_dice_proxy'],
    'I2': ['s1_s2_intersection_voxels'],
    'I3': ['s2_outside_s1_fraction','s1_outside_s2_fraction'],
}

# Best config drops
BEST_DROPS = {
    'G1','G2','G3','G4','I1','I2','I3'
}

# Priority subgroups for this ablation
PRIORITY = {'D3','D4','E2','E3','F1','H1','H2','G1','G2','G3','I1','I2','I3'}


def micro_dice(t, p, g): return 2.0 * t / max(1.0, p + g)


def dinkelbach(s1t, s1p, s2t, s2p, g, max_iter=50, tol=1e-8):
    lam = 2.0 * s1t.sum() / max(1.0, s1p.sum() + g.sum())
    dt, dp = s2t - s1t, s2p - s1p
    for _ in range(max_iter):
        use = 2.0 * dt - lam * dp > 0
        t = np.where(use, s2t, s1t).sum(); p = np.where(use, s2p, s1p).sum()
        nl = 2.0 * t / max(1.0, p + g.sum())
        if abs(nl - lam) < tol: break
        lam = nl
    return lam, (2.0 * dt - lam * dp > 0).astype(int)


def scan_threshold(probs, s1t, s1p, s2t, s2p, g):
    bt, bd = 0.5, 0.0
    for th in np.arange(0.05, 0.96, 0.05):
        u = probs >= th
        if u.sum() == 0: d = micro_dice(float(s1t.sum()), float(s1p.sum()), float(g.sum()))
        elif u.sum() == len(u): d = micro_dice(float(s2t.sum()), float(s2p.sum()), float(g.sum()))
        else:
            tp = np.where(u, s2t, s1t).sum(); pd = np.where(u, s2p, s1p).sum()
            d = micro_dice(tp, pd, g.sum())
        if d > bd: bd, bt = d, float(th)
    return bt, bd


def run_cv(feature_cols, rows, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt):
    cat_in = [c for c in CAT_COLS if c in feature_cols]
    num_in = [c for c in feature_cols if c not in cat_in]
    nc = len(cat_in)

    X = np.empty((len(rows), nc + len(num_in)), dtype=object)
    for i in range(nc): X[:, i] = [str(r.get(cat_in[i], '')) for r in rows]
    for j in range(len(num_in)): X[:, nc + j] = np.array(
        [float(r.get(num_in[j], 0) or 0) for r in rows], dtype=np.float64)

    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(range(nc))),
        ('num', StandardScaler(), list(range(nc, nc + len(num_in)))),
    ])

    gkf = GroupKFold(n_splits=5)
    fold_dice = []; all_tp = []; all_pred_m = []; all_gt = []; total_fp = 0; total_fn = 0; total_pred_s2 = 0

    for tr, te in gkf.split(X, groups=groups):
        Xtr, Xte = prep.fit_transform(X[tr]), prep.transform(X[te])
        ts1t, ts1p = s1_tp[tr], s1_pred[tr]; ts2t, ts2p = s2_tp[tr], s2_pred[tr]; tg_ = gt[tr]
        es1t, es1p = s1_tp[te], s1_pred[te]; es2t, es2p = s2_tp[te], s2_pred[te]; eg_ = gt[te]
        lam_f, lab_train = dinkelbach(ts1t, ts1p, ts2t, ts2p, tg_)
        _, lab_test = dinkelbach(es1t, es1p, es2t, es2p, eg_)

        model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=5,
                                        class_weight="balanced", random_state=42, n_jobs=-1)
        model.fit(np.asarray(Xtr), lab_train)
        prob_tr = model.predict_proba(np.asarray(Xtr))[:, 1]
        bth, _ = scan_threshold(prob_tr, ts1t, ts1p, ts2t, ts2p, tg_)
        prob_te = model.predict_proba(np.asarray(Xte))[:, 1]
        pred_f = (prob_te >= bth).astype(int)

        tp_sum = np.where(pred_f == 1, es2t, es1t).sum()
        pd_sum = np.where(pred_f == 1, es2p, es1p).sum()
        gt_sum = eg_.sum()
        fold_dice.append(micro_dice(tp_sum, pd_sum, gt_sum))
        all_tp.append(tp_sum); all_pred_m.append(pd_sum); all_gt.append(gt_sum)
        total_fp += int(((pred_f == 1) & (lab_test == 0)).sum())
        total_fn += int(((pred_f == 0) & (lab_test == 1)).sum())
        total_pred_s2 += int(pred_f.sum())

    pooled = micro_dice(sum(all_tp), sum(all_pred_m), sum(all_gt))
    return {
        "pooled_OOF": round(pooled, 4),
        "fold_mean": round(float(np.mean(fold_dice)), 4),
        "fold_std": round(float(np.std(fold_dice)), 4),
        "FP": total_fp, "FN": total_fn, "total_pred_s2": total_pred_s2,
    }


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    cols, rows = [], []
    with open(CSV_PATH, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): cols = list(r.keys()); rows.append(r)
    all_feat = [c for c in cols if c.startswith('feature__')]
    groups = np.array([r['meta__case_name'] for r in rows])
    s1_tp = np.array([float(r.get('target__s1_tp', 0)) for r in rows], dtype=np.float64)
    s1_pred = np.array([float(r.get('target__s1_pred_voxels', 0)) for r in rows], dtype=np.float64)
    s2_tp = np.array([float(r.get('target__s2_raw_tp', 0)) for r in rows], dtype=np.float64)
    s2_pred = np.array([float(r.get('target__s2_raw_pred_voxels', 0)) for r in rows], dtype=np.float64)
    gt = np.array([float(r.get('target__gt_voxels', 0)) for r in rows], dtype=np.float64)
    always_s1 = micro_dice(float(s1_tp.sum()), float(s1_pred.sum()), float(gt.sum()))

    # Build feature name -> subgroup mapping
    feat_to_sg = {}
    for sg, names in ALL_SUBGROUPS.items():
        for n in names:
            feat_to_sg['feature__' + n] = sg

    # Baseline A: all features
    print("=== Baselines ===")
    r_all = run_cv(all_feat, rows, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt)
    base_a = r_all["pooled_OOF"]
    print(f"Baseline A (all {len(all_feat)} features): {base_a:.4f}  FP={r_all['FP']}  FN={r_all['FN']}  predS2={r_all['total_pred_s2']}")

    # Baseline B: no_ROI_no_overlap
    best_feat = all_feat
    for sg_name in BEST_DROPS:
        names = ALL_SUBGROUPS[sg_name]
        best_feat = [c for c in best_feat if c not in ['feature__' + n for n in names]]
    r_best = run_cv(best_feat, rows, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt)
    base_b = r_best["pooled_OOF"]
    print(f"Baseline B (best {len(best_feat)} features): {base_b:.4f}  FP={r_best['FP']}  FN={r_best['FN']}  predS2={r_best['total_pred_s2']}")

    results = {"always_s1": round(always_s1, 4), "baseline_A": r_all, "baseline_B": r_best,
               "delete_from_all": {}, "add_to_best": {}}

    # ── Exp 1: Delete ONE subgroup from ALL features ──
    print(f"\n=== Delete-one from ALL (baseline {base_a:.4f}) ===")
    print(f"{'Delete':>12s} {'pooled_OOF':>10s} {'Delta':>8s} {'FP':>6s} {'FN':>6s} {'predS2':>6s}")
    print("-" * 52)
    for sg_name in sorted(PRIORITY):
        names = ALL_SUBGROUPS[sg_name]
        kept = [c for c in all_feat if c not in ['feature__' + n for n in names]]
        r = run_cv(kept, rows, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt)
        delta = r["pooled_OOF"] - base_a
        results["delete_from_all"][sg_name] = r
        results["delete_from_all"][sg_name]["delta"] = round(delta, 4)
        print(f"  -{sg_name:<9s} {r['pooled_OOF']:>10.4f} {delta:>+8.4f} {r['FP']:>6d} {r['FN']:>6d} {r['total_pred_s2']:>6d}")

    # ── Exp 2: Add ONE subgroup to BEST features ──
    print(f"\n=== Add-one to BEST (baseline {base_b:.4f}) ===")
    print(f"{'Add':>12s} {'pooled_OOF':>10s} {'Delta':>8s} {'FP':>6s} {'FN':>6s} {'predS2':>6s}")
    print("-" * 52)
    for sg_name in sorted(PRIORITY & BEST_DROPS):  # only ROI/overlap groups
        add_feats = ['feature__' + n for n in ALL_SUBGROUPS[sg_name]]
        kept = best_feat + add_feats
        r = run_cv(kept, rows, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt)
        delta = r["pooled_OOF"] - base_b
        results["add_to_best"][sg_name] = r
        results["add_to_best"][sg_name]["delta"] = round(delta, 4)
        print(f"  +{sg_name:<9s} {r['pooled_OOF']:>10.4f} {delta:>+8.4f} {r['FP']:>6d} {r['FN']:>6d} {r['total_pred_s2']:>6d}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
