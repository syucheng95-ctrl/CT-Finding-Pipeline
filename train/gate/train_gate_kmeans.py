"""KMeans cluster features for gate training.

Key rules:
- Cluster per-fold (fit on train, predict on test) — no CV leak
- Cluster only from "state features" (no GT)
- Cluster ID one-hot added as extra RF features

Experiments:
  E0: Baseline RF + no_ROI_no_overlap
  E1: KMeans cluster only → RF
  E2: Baseline features + KMeans cluster → RF
  E3: Baseline features + KMeans distance-to-centroids → RF
"""

import csv, json, warnings, numpy as np
from pathlib import Path
import sklearn; sklearn.set_config(transform_output='default')
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "gate_training" / "outputs" / "gate_training_table.csv"
OUT_JSON = ROOT / "gate_training" / "outputs" / "gate_kmeans_results.json"

CAT_COLS = ['feature__pred_category','feature__anatomy_target','feature__laterality',
            'feature__anatomy_group','feature__final_policy','feature__final_tightness','feature__expert']

# Best feature config (no ROI, no overlap)
BEST_DROP = ("feature__roi_","feature__n_rois","feature__n_coarse_rois",
             "feature__n_proposal_rois","feature__roi_coarse_fraction",
             "feature__s1_s2_","feature__s2_overlap","feature__s1_covered",
             "feature__s2_outside","feature__s1_outside")

# State features for clustering (describing S2 vs S1 behavior, ROI quality, verifier confidence)
CLUSTER_FEATURES = [
    "feature__s2_s1_volume_ratio",
    "feature__s2_raw_log_pred_voxels",
    "feature__s1_pred_log_voxels",
    "feature__s2_roi_pred_density",
    "feature__n_rois",
    "feature__roi_total_volume",
    "feature__roi_max_volume",
    "feature__verified_ratio",
    "feature__verified_fg_ratio_max",
    "feature__verified_max_prob_max",
    "feature__proposal_max_volume",
    "feature__proposal_mean_volume",
    "feature__n_verified",
    "feature__n_rejected",
]
N_CLUSTERS = 5


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


def build_X(rows, all_feat, drop_prefixes):
    """Build feature matrix from CSV rows."""
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
    return X, prep, kept


def build_cluster_X(rows, features_list):
    """Build cluster feature matrix (only continuous state features)."""
    present = [c for c in features_list if c in rows[0]]
    arr = np.array([[float(r.get(c, 0) or 0) for c in present] for r in rows], dtype=np.float64)
    # Fill NaN with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, present


def one_hot(cluster_ids, n_clusters):
    oh = np.zeros((len(cluster_ids), n_clusters), dtype=np.float64)
    oh[np.arange(len(cluster_ids)), cluster_ids] = 1.0
    return oh


def run_cv(name, X, prep, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt,
           cluster_X=None):
    """Run 5-fold CV. cluster_X is used to fit KMeans per-fold."""
    gkf = GroupKFold(n_splits=5)
    fold_dice = []; all_tp = []; all_pred_m = []; all_gt = []
    total_fp = 0; total_fn = 0; total_pred_s2 = 0
    cluster_ids_all = np.full(len(groups), -1, dtype=int)

    for tr, te in gkf.split(X, groups=groups):
        Xtr, Xte = prep.fit_transform(X[tr]), prep.transform(X[te])
        ts1t, ts1p = s1_tp[tr], s1_pred[tr]; ts2t, ts2p = s2_tp[tr], s2_pred[tr]; tg_ = gt[tr]
        es1t, es1p = s1_tp[te], s1_pred[te]; es2t, es2p = s2_tp[te], s2_pred[te]; eg_ = gt[te]

        lam_f, lab_train = dinkelbach(ts1t, ts1p, ts2t, ts2p, tg_)
        _, lab_test = dinkelbach(es1t, es1p, es2t, es2p, eg_)

        Xtr_a, Xte_a = np.asarray(Xtr), np.asarray(Xte)

        # ── Per-fold KMeans ──
        if cluster_X is not None:
            cXtr, cXte = cluster_X[tr], cluster_X[te]
            # Standardize cluster features on train
            cs = StandardScaler()
            cXtr_s = cs.fit_transform(cXtr)
            cXte_s = cs.transform(cXte)
            km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
            km.fit(cXtr_s)
            cid_tr = km.predict(cXtr_s)
            cid_te = km.predict(cXte_s)
            cluster_ids_all[te] = cid_te

            if "clusterID" in name:
                oh_tr = one_hot(cid_tr, N_CLUSTERS)
                oh_te = one_hot(cid_te, N_CLUSTERS)
                Xtr_a = np.hstack([Xtr_a, oh_tr])
                Xte_a = np.hstack([Xte_a, oh_te])
            elif "clusterDist" in name:
                dist_tr = km.transform(cXtr_s)
                dist_te = km.transform(cXte_s)
                Xtr_a = np.hstack([Xtr_a, dist_tr])
                Xte_a = np.hstack([Xte_a, dist_te])

        model = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)
        model.fit(Xtr_a, lab_train)

        prob_tr = model.predict_proba(Xtr_a)[:, 1]
        bth, _ = scan_threshold(prob_tr, ts1t, ts1p, ts2t, ts2p, tg_)
        prob_te = model.predict_proba(Xte_a)[:, 1]
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
        "fold_dice": [round(float(v), 4) for v in fold_dice],
        "fold_mean": round(float(np.mean(fold_dice)), 4),
        "fold_std": round(float(np.std(fold_dice)), 4),
        "pooled_OOF": round(pooled, 4),
        "FP": total_fp, "FN": total_fn,
        "total_pred_s2": total_pred_s2,
    }, cluster_ids_all


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

    # Build main feature matrix
    X, prep, kept_main = build_X(rows, all_feat, BEST_DROP)
    # Build cluster feature matrix
    cluster_X, cluster_cols = build_cluster_X(rows, CLUSTER_FEATURES)
    print(f"Main features: {len(kept_main)}, Cluster features: {len(cluster_cols)}")
    print(f"Cluster cols: {cluster_cols}")
    print(f"Always S1: {always_s1:.4f}\n")

    all_results = {"always_s1": round(always_s1, 4), "experiments": {}}

    experiments = {
        "E0_baseline": {"use_cluster": False, "cluster_mode": None},
        "E2_clusterID": {"use_cluster": True, "cluster_mode": "clusterID"},
        "E3_clusterDist": {"use_cluster": True, "cluster_mode": "clusterDist"},
    }

    print(f"{'Experiment':<20s} {'fold_mean':>10s} {'pooled_OOF':>10s} {'FP':>6s} {'FN':>6s} {'predS2':>6s}")
    print("-" * 62)

    for exp_name, cfg in experiments.items():
        if cfg["use_cluster"]:
            mode = cfg["cluster_mode"]
            r = run_cv(mode, X, prep, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt,
                       cluster_X=cluster_X)[0]
        else:
            r = run_cv("baseline", X, prep, groups, s1_tp, s1_pred, s2_tp, s2_pred, gt,
                       cluster_X=None)[0]
        all_results["experiments"][exp_name] = r
        print(f"{exp_name:<20s} {r['fold_mean']:>10.4f} {r['pooled_OOF']:>10.4f} {r['FP']:>6d} {r['FN']:>6d} {r['total_pred_s2']:>6d}")

    # ── E4: cluster per category (2a/2b/2g only) ──
    # For E4, we take baseline features + cluster only for specific categories
    # Actually simpler: add cluster features only for rows where category is 2a/2b/2g
    # For other categories, cluster features are zero

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n-> {OUT_JSON}")

    best_val, best_key = 0.0, ""
    for k, v in all_results["experiments"].items():
        if v["pooled_OOF"] > best_val:
            best_val, best_key = v["pooled_OOF"], k
    print(f"Best: {best_key} = {best_val:.4f} (vs baseline 0.2373)")


if __name__ == "__main__":
    main()
