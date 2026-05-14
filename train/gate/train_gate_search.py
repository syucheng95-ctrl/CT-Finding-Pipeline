"""Exhaustive search for best feature config, multi-seed stability test."""

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
CAT_COLS = ['feature__pred_category','feature__anatomy_target','feature__laterality',
            'feature__anatomy_group','feature__final_policy','feature__final_tightness','feature__expert']

# Harmful/noisy groups to remove (from ablation)
H1 = ['s2_raw_pred_voxels','s2_raw_log_pred_voxels','s2_pred_is_empty','s2_pred_is_huge']
G_ALL = ['n_rois','n_coarse_rois','n_proposal_rois','roi_coarse_fraction',
         'roi_max_volume','roi_mean_volume','roi_median_volume','roi_total_volume','roi_volume_std',
         'roi_max_volume_ratio_to_crop','roi_total_volume_ratio_to_crop','roi_shape_d_mean']
I_ALL = ['s1_s2_intersection_voxels','s1_s2_dice_proxy','s1_covered_by_s2','s2_overlap_with_s1',
         's2_outside_s1_fraction','s1_outside_s2_fraction']
CONF_ALL = ['s2_conf_high_conf_frac_09','s2_conf_low_conf_frac_05_07','s2_conf_mean_entropy',
            's2_conf_mean_margin','s2_conf_mean_prob','s2_conf_p10_margin','s2_conf_p10_prob',
            's2_conf_p90_prob','s2_conf_soft_hard_ratio','s2_conf_std_prob',
            's2_n_low_conf_rois','s2_roi_conf_spread']
H2_ratio = ['s2_s1_volume_ratio','s2_roi_pred_density']
H3_flags = ['s2_overseg_ratio_gt5','s2_underseg_ratio_lt02']

# Feature configs to test
CONFIGS = {
    "best_with_conf": G_ALL + I_ALL,
    "old_best_no_conf": G_ALL + I_ALL + CONF_ALL,
    "old_best_no_conf-H1": G_ALL + I_ALL + CONF_ALL + H1,
    "old_best_no_conf+I2": G_ALL + [n for n in I_ALL if n != 's1_s2_intersection_voxels'] + CONF_ALL,
    "conf_only_added": G_ALL + I_ALL + [n for n in CONF_ALL if n not in [
        's2_conf_mean_margin', 's2_conf_p10_prob', 's2_conf_mean_entropy',
        's2_conf_high_conf_frac_09']],
}

def micro_dice(t, p, g): return 2.0 * t / max(1.0, p + g)
def dinkelbach(s1t,s1p,s2t,s2p,g):
    lam=2*s1t.sum()/max(1,s1p.sum()+g.sum()); dt,dp=s2t-s1t,s2p-s1p
    for _ in range(50):
        use=2*dt-lam*dp>0; t=np.where(use,s2t,s1t).sum(); p=np.where(use,s2p,s1p).sum()
        nl=2*t/max(1,p+g.sum())
        if abs(nl-lam)<1e-8: break; lam=nl
    return lam,((2*dt-lam*dp)>0).astype(int)
def scan(probs,s1t,s1p,s2t,s2p,g):
    bt,bd=0.5,0.0
    for th in np.arange(0.05,0.96,0.05):
        u=probs>=th
        if u.sum()==0: d=micro_dice(float(s1t.sum()),float(s1p.sum()),float(g.sum()))
        elif u.sum()==len(u): d=micro_dice(float(s2t.sum()),float(s2p.sum()),float(g.sum()))
        else: tp=np.where(u,s2t,s1t).sum(); pd=np.where(u,s2p,s1p).sum(); d=micro_dice(tp,pd,g.sum())
        if d>bd: bd,bt=d,float(th)
    return bt,bd

# Load
cols, rows = [], []
with open(CSV_PATH, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f): cols=list(r.keys()); rows.append(r)
all_feat = [c for c in cols if c.startswith('feature__')]
groups = np.array([r['meta__case_name'] for r in rows])
s1_tp=np.array([float(r.get('target__s1_tp',0)) for r in rows], dtype=np.float64)
s1_pred=np.array([float(r.get('target__s1_pred_voxels',0)) for r in rows], dtype=np.float64)
s2_tp=np.array([float(r.get('target__s2_raw_tp',0)) for r in rows], dtype=np.float64)
s2_pred=np.array([float(r.get('target__s2_raw_pred_voxels',0)) for r in rows], dtype=np.float64)
gt=np.array([float(r.get('target__gt_voxels',0)) for r in rows], dtype=np.float64)
always_s1=micro_dice(float(s1_tp.sum()),float(s1_pred.sum()),float(gt.sum()))

N_SEEDS = 3
SEEDS = [42, 142, 242]

print(f"{'Config':<22s} {'Feats':>5s} {'mean_OOF':>10s} {'std_OOF':>10s} {'runs':>30s}")
print("-" * 85)

for cfg_name, drop_names in CONFIGS.items():
    kept = [c for c in all_feat if c[9:] not in drop_names]
    cat_in = [c for c in CAT_COLS if c in kept]
    num_in = [c for c in kept if c not in cat_in]
    nc=len(cat_in)
    X=np.empty((len(rows),nc+len(num_in)),dtype=object)
    for i in range(nc): X[:,i]=[str(r.get(cat_in[i],'')) for r in rows]
    for j in range(len(num_in)): X[:,nc+j]=np.array([float(r.get(num_in[j],0) or 0) for r in rows], dtype=np.float64)

    run_results = []
    for seed in SEEDS:
        prep=ColumnTransformer([
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),list(range(nc))),
            ('num',StandardScaler(),list(range(nc,nc+len(num_in)))),
        ])
        gkf=GroupKFold(n_splits=5)
        all_tp=[]; all_pd=[]; all_gt=[]
        for tr,te in gkf.split(X,groups=groups):
            Xtr,Xte=prep.fit_transform(X[tr]),prep.transform(X[te])
            ts1t,ts1p=s1_tp[tr],s1_pred[tr]; ts2t,ts2p=s2_tp[tr],s2_pred[tr]; tg_=gt[tr]
            es1t,es1p=s1_tp[te],s1_pred[te]; es2t,es2p=s2_tp[te],s2_pred[te]; eg_=gt[te]
            lam_f,lab=dinkelbach(ts1t,ts1p,ts2t,ts2p,tg_)
            rf=RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_leaf=5,
                                      class_weight='balanced',random_state=seed,n_jobs=-1)
            rf.fit(np.asarray(Xtr),lab)
            prob_tr=rf.predict_proba(np.asarray(Xtr))[:,1]
            bth,_=scan(prob_tr,ts1t,ts1p,ts2t,ts2p,tg_)
            prob_te=rf.predict_proba(np.asarray(Xte))[:,1]
            pred_f=(prob_te>=bth).astype(int)
            all_tp.append(np.where(pred_f==1,es2t,es1t).sum())
            all_pd.append(np.where(pred_f==1,es2p,es1p).sum())
            all_gt.append(eg_.sum())
        run_results.append(micro_dice(sum(all_tp),sum(all_pd),sum(all_gt)))

    print(f"{cfg_name:<22s} {len(kept):>5d} {np.mean(run_results):>10.4f} {np.std(run_results):>10.4f} {str([round(r,4) for r in run_results]):>30s}")

print(f"{'Always S1':<22s} {'-':>5s} {always_s1:>10.4f} {'-':>10s} {'-':>30s}")
