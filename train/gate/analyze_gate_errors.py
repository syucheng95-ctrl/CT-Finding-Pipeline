"""Analyze gate FP/FN per finding across configurations.

Outputs a table: per finding, per config, was it FP or FN (or correct).
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
OUT_CSV = ROOT / "gate_training" / "outputs" / "gate_error_analysis.csv"

CAT_COLS = ['feature__pred_category','feature__anatomy_target','feature__laterality',
            'feature__anatomy_group','feature__final_policy','feature__final_tightness','feature__expert']

CONFIGS = {
    "no_ROI": {
        "drop": ("feature__roi_", "feature__n_rois", "feature__n_coarse_rois",
                 "feature__n_proposal_rois", "feature__roi_coarse_fraction"),
    },
    "no_ROI_no_overlap": {
        "drop": ("feature__roi_", "feature__n_rois", "feature__n_coarse_rois",
                 "feature__n_proposal_rois", "feature__roi_coarse_fraction",
                 "feature__s1_s2_", "feature__s2_overlap", "feature__s1_covered",
                 "feature__s2_outside", "feature__s1_outside"),
    },
    "all_features": {
        "drop": (),
    },
    "all_plus_overlap": {
        "drop": (),  # overlap is already in CSV
    },
}

def micro_dice(t, p, g): return (2.0*t)/max(1.0,p+g)
def dinkelbach(s1t,s1p,s2t,s2p,g):
    lam=(2*s1t.sum())/max(1,s1p.sum()+g.sum()); dt,dp=s2t-s1t,s2p-s1p
    for _ in range(50):
        use=2*dt-lam*dp>0; t=np.where(use,s2t,s1t).sum(); p=np.where(use,s2p,s1p).sum()
        nl=(2*t)/max(1,p+g.sum())
        if abs(nl-lam)<1e-8: break
        lam=nl
    return lam,((2*dt-lam*dp)>0).astype(int)

def scan_thresh(probs,s1t,s1p,s2t,s2p,g):
    bt,bd=0.5,0.0
    for th in np.arange(0.05,0.96,0.05):
        u=probs>=th
        if u.sum()==0: d=micro_dice(float(s1t.sum()),float(s1p.sum()),float(g.sum()))
        elif u.sum()==len(u): d=micro_dice(float(s2t.sum()),float(s2p.sum()),float(g.sum()))
        else:
            tp=np.where(u,s2t,s1t).sum(); pd=np.where(u,s2p,s1p).sum(); gs=g.sum()
            d=micro_dice(tp,pd,gs)
        if d>bd: bd,bt=d,float(th)
    return bt,bd

def main():
    cols, rows = [], []
    with open(CSV_PATH, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): cols=list(r.keys()); rows.append(r)
    all_feat = [c for c in cols if c.startswith('feature__')]
    groups_arr = np.array([r['meta__case_name'] for r in rows])
    fids = np.array([r['meta__finding_id'] for r in rows])
    cats = np.array([r['meta__category'] for r in rows])

    s1_tp=np.array([float(r.get('target__s1_tp',0)) for r in rows], dtype=np.float64)
    s1_pred=np.array([float(r.get('target__s1_pred_voxels',0)) for r in rows], dtype=np.float64)
    s2_tp=np.array([float(r.get('target__s2_raw_tp',0)) for r in rows], dtype=np.float64)
    s2_pred=np.array([float(r.get('target__s2_raw_pred_voxels',0)) for r in rows], dtype=np.float64)
    gt=np.array([float(r.get('target__gt_voxels',0)) for r in rows], dtype=np.float64)

    # Global oracle for reference
    _, oracle_global = dinkelbach(s1_tp, s1_pred, s2_tp, s2_pred, gt)
    print(f"Oracle global: {oracle_global.sum()}/{len(oracle_global)} use_s2=1")

    # Per-finding accumulator: finding_id -> list of (config, fold, pred, oracle_fold)
    errors_by_fid = {r['meta__finding_id']: [] for r in rows}

    gkf = GroupKFold(n_splits=5)

    for config_name, cfg in CONFIGS.items():
        drop_prefixes = cfg["drop"]
        kept = [c for c in all_feat if not c.startswith(drop_prefixes)]

        cat_in = [c for c in CAT_COLS if c in kept]
        num_in = [c for c in kept if c not in cat_in]
        nc = len(cat_in)
        X = np.empty((len(rows), nc+len(num_in)), dtype=object)
        for i in range(nc): X[:,i] = [str(r.get(cat_in[i],'')) for r in rows]
        for j in range(len(num_in)): X[:,nc+j] = np.array([float(r.get(num_in[j],0) or 0) for r in rows], dtype=np.float64)

        prep = ColumnTransformer([
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),list(range(nc))),
            ('num',StandardScaler(),list(range(nc,nc+len(num_in)))),
        ])

        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({len(kept)} features)")

        fold_dice = []
        for fold_i, (tr, te) in enumerate(gkf.split(X, groups=groups_arr)):
            Xtr, Xte = prep.fit_transform(X[tr]), prep.transform(X[te])
            ts1t,ts1p=s1_tp[tr],s1_pred[tr]; ts2t,ts2p=s2_tp[tr],s2_pred[tr]; tg=gt[tr]
            es1t,es1p=s1_tp[te],s1_pred[te]; es2t,es2p=s2_tp[te],s2_pred[te]; eg=gt[te]

            lam_f, lab_f = dinkelbach(ts1t,ts1p,ts2t,ts2p,tg)
            rf=RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_leaf=5,
                                      class_weight='balanced',random_state=42,n_jobs=-1)
            rf.fit(np.asarray(Xtr), lab_f)
            prob_tr=rf.predict_proba(np.asarray(Xtr))[:,1]
            bth,_=scan_thresh(prob_tr,ts1t,ts1p,ts2t,ts2p,tg)
            prob_te=rf.predict_proba(np.asarray(Xte))[:,1]
            pred_f=(prob_te>=bth).astype(int)

            # Per-fold oracle label for comparison
            _, lab_test = dinkelbach(es1t,es1p,es2t,es2p,eg)

            tp_sum=np.where(pred_f==1,es2t,es1t).sum()
            pd_sum=np.where(pred_f==1,es2p,es1p).sum()
            dice=micro_dice(tp_sum,pd_sum,eg.sum())
            fold_dice.append(dice)
            print(f"  Fold {fold_i+1}: dice={dice:.4f} thresh={bth:.2f} n_pred_s2={pred_f.sum()}")

            # Record per-finding errors
            for i, idx in enumerate(te):
                fid = fids[idx]
                errors_by_fid[fid].append({
                    "config": config_name,
                    "fold": fold_i+1,
                    "pred": int(pred_f[i]),
                    "oracle": int(lab_test[i]),
                    "prob": round(float(prob_te[i]), 4),
                })

        print(f"  Mean: {np.mean(fold_dice):.4f} +-{np.std(fold_dice):.4f}")

    # ── Write per-finding error table ──
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = ['finding_id', 'category', 'oracle_global']
        config_names = list(CONFIGS.keys())
        for cn in config_names:
            header += [f'{cn}_FP_count', f'{cn}_FN_count',
                       f'{cn}_pred_mean', f'{cn}_prob_mean']
        writer.writerow(header)

        for fid in sorted(errors_by_fid):
            entries = errors_by_fid[fid]
            cat = cats[list(fids).index(fid)]
            og = int(oracle_global[list(fids).index(fid)])
            row = [fid, cat, og]
            for cn in config_names:
                e = [x for x in entries if x['config'] == cn]
                fp = sum(1 for x in e if x['pred']==1 and x['oracle']==0)
                fn = sum(1 for x in e if x['pred']==0 and x['oracle']==1)
                pred_mean = np.mean([x['pred'] for x in e])
                prob_mean = np.mean([x['prob'] for x in e])
                row += [fp, fn, round(pred_mean,2), round(prob_mean,4)]
            writer.writerow(row)

    print(f"\nError analysis -> {OUT_CSV}")

    # ── Summary stats ──
    print(f"\n{'='*70}")
    print("Per-config summary: FP (pred S2 but oracle says S1), FN (pred S1 but oracle says S2)")
    print(f"{'Config':<25s} {'TotalFP':>8s} {'TotalFN':>8s} {'FPrate':>8s} {'FNrate':>8s}")
    print("-"*60)
    for cn in config_names:
        all_fp = 0; all_fn = 0
        for fid in errors_by_fid:
            e = [x for x in errors_by_fid[fid] if x['config']==cn]
            all_fp += sum(1 for x in e if x['pred']==1 and x['oracle']==0)
            all_fn += sum(1 for x in e if x['pred']==0 and x['oracle']==1)
        total = len(rows)*5
        fp_rate = all_fp/max(1,total)
        fn_rate = all_fn/max(1,total)
        print(f"{cn:<25s} {all_fp:>8d} {all_fn:>8d} {fp_rate:>8.3f} {fn_rate:>8.3f}")


if __name__ == '__main__':
    main()
