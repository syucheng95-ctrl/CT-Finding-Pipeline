"""V3: C regressor variants.

C1: RF regressor + normalized gain + quantile threshold
C2: RF regressor + normalized gain + top-k selection
C3: HistGradientBoostingRegressor + normalized gain + top-k
"""

import csv, json, math, warnings, numpy as np
from pathlib import Path
import sklearn; sklearn.set_config(transform_output='default')
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "gate_training" / "outputs" / "gate_training_table.csv"
OUT_JSON = ROOT / "gate_training" / "outputs" / "gate_v3_results.json"

CAT_COLS = ['feature__pred_category','feature__anatomy_target','feature__laterality',
            'feature__anatomy_group','feature__final_policy','feature__final_tightness','feature__expert']

ROI_DROP = ("feature__roi_","feature__n_rois","feature__n_coarse_rois",
            "feature__n_proposal_rois","feature__roi_coarse_fraction")
OVERLAP_DROP = ("feature__s1_s2_","feature__s2_overlap","feature__s1_covered",
                "feature__s2_outside","feature__s1_outside")

FEATURE_CONFIGS = {
    "all": (),
    "no_ROI": ROI_DROP,
    "no_overlap": OVERLAP_DROP,
    "no_ROI_no_overlap": ROI_DROP + OVERLAP_DROP,
}


def micro_dice(t, p, g): return 2.0 * t / max(1.0, p + g)


def dinkelbach(s1t, s1p, s2t, s2p, g, max_iter=50, tol=1e-8):
    lam = 2.0 * s1t.sum() / max(1.0, s1p.sum() + g.sum())
    dt, dp = s2t - s1t, s2p - s1p
    for _ in range(max_iter):
        use = 2.0 * dt - lam * dp > 0
        t = np.where(use, s2t, s1t).sum()
        p = np.where(use, s2p, s1p).sum()
        nl = 2.0 * t / max(1.0, p + g.sum())
        if abs(nl - lam) < tol: break
        lam = nl
    gain = 2.0 * dt - lam * dp
    gain_norm = gain / np.maximum(1.0, g + s1p)
    return lam, gain_norm, gain, (gain > 0).astype(int)


def apply_and_dice(use_s2, s1t, s1p, s2t, s2p, g):
    tp = np.where(use_s2, s2t, s1t).sum()
    pd = np.where(use_s2, s2p, s1p).sum()
    return micro_dice(tp, pd, g.sum()), tp, pd, g.sum()


def scan_quantile_threshold(pred_gain_train, s1t, s1p, s2t, s2p, g, n_steps=101):
    qs = np.quantile(pred_gain_train, np.linspace(0, 1, n_steps))
    best_th, best_d = 0.0, 0.0
    for th in qs:
        d = apply_and_dice(pred_gain_train > th, s1t, s1p, s2t, s2p, g)[0]
        if d > best_d: best_d, best_th = d, float(th)
    return best_th, best_d


def scan_topk_fraction(pred_gain_train, s1t, s1p, s2t, s2p, g, max_frac=0.25):
    order = np.argsort(-pred_gain_train)
    max_k = min(len(order), int(round(len(order) * max_frac)))
    best_k, best_frac, best_d = 0, 0.0, 0
    for k in range(0, max_k + 1):
        if k == 0:
            d = micro_dice(float(s1t.sum()), float(s1p.sum()), float(g.sum()))
        else:
            idx = order[:k]
            use_s2 = np.zeros(len(s1t), dtype=bool)
            use_s2[idx] = True
            d = apply_and_dice(use_s2, s1t, s1p, s2t, s2p, g)[0]
        if d > best_d:
            best_d, best_k = d, k
            best_frac = k / max(1, len(order))
    return best_frac, best_k, best_d


def run_experiment(strategy_name, model_factory, feat_config_name, drop_prefixes,
                   all_feat, rows, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt):
    kept = [c for c in all_feat if not c.startswith(drop_prefixes)]
    cat_in = [c for c in CAT_COLS if c in kept]
    num_in = [c for c in kept if c not in cat_in]
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
    fold_dice = []; all_tp = []; all_pred_m = []; all_gt = []
    total_fp = 0; total_fn = 0; total_pred_s2 = 0

    for tr, te in gkf.split(X, groups=groups):
        Xtr, Xte = prep.fit_transform(X[tr]), prep.transform(X[te])
        ts1t, ts1p = s1_tp[tr], s1_pred[tr]; ts2t, ts2p = s2_tp[tr], s2_pred[tr]; tg_ = gt[tr]
        es1t, es1p = s1_tp[te], s1_pred[te]; es2t, es2p = s2_tp[te], s2_pred[te]; eg_ = gt[te]

        lam_f, gain_norm_train, gain_vox_train, lab_train = dinkelbach(ts1t, ts1p, ts2t, ts2p, tg_)
        _, gain_norm_test, gain_vox_test, lab_test = dinkelbach(es1t, es1p, es2t, es2p, eg_)

        Xtr_a, Xte_a = np.asarray(Xtr), np.asarray(Xte)
        model = model_factory()
        model.fit(Xtr_a, gain_norm_train)
        pred_tr = model.predict(Xtr_a)
        pred_te = model.predict(Xte_a)

        if strategy_name in ("C1_quantile",):
            best_th, _ = scan_quantile_threshold(pred_tr, ts1t, ts1p, ts2t, ts2p, tg_)
            pred_f = (pred_te > best_th).astype(int)
            best_k = -1
        elif strategy_name in ("C2_topk", "C3_hgb_topk"):
            best_frac, best_k_train, _ = scan_topk_fraction(pred_tr, ts1t, ts1p, ts2t, ts2p, tg_)
            order = np.argsort(-pred_te)
            pred_f = np.zeros(len(te), dtype=int)
            best_k = int(round(best_frac * len(te)))
            if best_k > 0:
                pred_f[order[:best_k]] = 1
            best_th = -1

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
        "n_features": len(kept),
        "fold_dice": [round(float(v), 4) for v in fold_dice],
        "fold_mean": round(float(np.mean(fold_dice)), 4),
        "fold_std": round(float(np.std(fold_dice)), 4),
        "pooled_OOF": round(pooled, 4),
        "pooled_tp": int(sum(all_tp)), "pooled_pred": int(sum(all_pred_m)), "pooled_gt": int(sum(all_gt)),
        "FP": total_fp, "FN": total_fn,
        "total_pred_s2": total_pred_s2,
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

    strategies = {
        "C1_quantile": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "C2_topk": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "C3_hgb_topk": lambda: HistGradientBoostingRegressor(
            max_depth=4, min_samples_leaf=5, learning_rate=0.05, max_iter=200, random_state=42),
    }

    all_results = {"always_s1": round(always_s1, 4), "results": {}}

    print(f"{'Strategy':<18s} {'Features':<20s} {'fold_mean':>10s} {'pooled_OOF':>10s} {'FP':>6s} {'FN':>6s} {'predS2':>6s}")
    print("-" * 80)
    print(f"{'Always S1':<18s} {'-':<20s} {'-':>10s} {always_s1:>10.4f} {'-':>6s} {'-':>6s} {'-':>6s}")

    for strat_name, make_model in strategies.items():
        all_results["results"][strat_name] = {}
        for fc_name, drop_pref in FEATURE_CONFIGS.items():
            r = run_experiment(strat_name, make_model, fc_name, drop_pref,
                              all_feat, rows, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt)
            all_results["results"][strat_name][fc_name] = r
            print(f"{strat_name:<18s} {fc_name:<20s} {r['fold_mean']:>10.4f} {r['pooled_OOF']:>10.4f} {r['FP']:>6d} {r['FN']:>6d} {r['total_pred_s2']:>6d}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n-> {OUT_JSON}")

    best_val, best_key = 0.0, ""
    for strat_name in strategies:
        for fc_name in FEATURE_CONFIGS:
            pooled = all_results["results"][strat_name][fc_name]["pooled_OOF"]
            if pooled > best_val:
                best_val, best_key = pooled, f"{strat_name}__{fc_name}"
    print(f"\nBest: {best_key} pooled_OOF = {best_val:.4f}")


if __name__ == "__main__":
    main()
