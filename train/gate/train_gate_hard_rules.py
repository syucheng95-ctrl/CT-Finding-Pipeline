"""Category-specific hard gate search.

For each GroupKFold fold, this script:
  1. builds candidate rules on train only;
  2. selects one rule per category by train gain;
  3. applies selected rules to test;
  4. reports pooled OOF Dice.

Rules use only feature__ columns. target__ columns are used only for selecting
rules on train and evaluating on test.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "gate_training" / "outputs" / "gate_training_table.csv"
OUT_JSON = ROOT / "gate_training" / "outputs" / "gate_hard_rules_results.json"

CANDIDATE_CATS = ["1c", "2a", "2b", "2g"]

# Features with current evidence or direct clinical/technical meaning.
RULE_FEATURES = [
    "feature__s2_s1_volume_ratio",
    "feature__s2_roi_pred_density",
    "feature__s2_overseg_ratio_gt5",
    "feature__s2_underseg_ratio_lt02",
    "feature__s1_pred_log_voxels",
    "feature__s1_pred_voxels",
    "feature__s2_raw_log_pred_voxels",
    "feature__s2_raw_pred_voxels",
    "feature__n_rois",
    "feature__n_coarse_rois",
    "feature__n_proposal_rois",
    "feature__roi_coarse_fraction",
    "feature__roi_max_volume",
    "feature__roi_total_volume",
    "feature__roi_max_volume_ratio_to_crop",
    "feature__s2_overlap_with_s1",
    "feature__s1_covered_by_s2",
    "feature__s1_s2_dice_proxy",
    "feature__s1_s2_intersection_voxels",
    "feature__s2_conf_mean_prob",
    "feature__s2_conf_p10_prob",
    "feature__s2_conf_p90_prob",
    "feature__s2_conf_high_conf_frac_09",
    "feature__s2_conf_low_conf_frac_05_07",
    "feature__s2_conf_mean_margin",
    "feature__s2_conf_p10_margin",
    "feature__s2_conf_mean_entropy",
    "feature__s2_conf_std_prob",
    "feature__s2_roi_conf_spread",
    "feature__s2_n_low_conf_rois",
    "feature__verified_ratio",
    "feature__verified_max_prob_max",
    "feature__verified_fg_ratio_max",
]


def as_float(row: dict, col: str) -> float:
    try:
        v = row.get(col, 0)
        if v is None or v == "":
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def micro_dice(tp: float, pred: float, gt: float) -> float:
    return 2.0 * tp / max(1.0, pred + gt)


def dinkelbach(s1_tp, s1_pred, s2_tp, s2_pred, gt, max_iter=50, tol=1e-8):
    lam = micro_dice(float(s1_tp.sum()), float(s1_pred.sum()), float(gt.sum()))
    dt = s2_tp - s1_tp
    dp = s2_pred - s1_pred
    for _ in range(max_iter):
        use = 2.0 * dt - lam * dp > 0
        tp = np.where(use, s2_tp, s1_tp).sum()
        pred = np.where(use, s2_pred, s1_pred).sum()
        new_lam = micro_dice(float(tp), float(pred), float(gt.sum()))
        if abs(new_lam - lam) < tol:
            break
        lam = new_lam
    return lam


def rule_gain(mask, lam, s1_tp, s1_pred, s2_tp, s2_pred):
    dt = s2_tp - s1_tp
    dp = s2_pred - s1_pred
    return float((2.0 * dt[mask] - lam * dp[mask]).sum())


def thresholds(values: np.ndarray) -> list[float]:
    vals = values[np.isfinite(values)]
    vals = vals[vals != 0]
    if len(vals) == 0:
        return []
    qs = np.unique(np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9]))
    extra = []
    # Common meaningful thresholds.
    if vals.min() <= 1.0 <= vals.max():
        extra.append(1.0)
    if vals.min() <= 0.5 <= vals.max():
        extra.append(0.5)
    if vals.min() <= 0.7 <= vals.max():
        extra.append(0.7)
    if vals.min() <= 0.9 <= vals.max():
        extra.append(0.9)
    return sorted(set(float(x) for x in list(qs) + extra))


def make_single_conditions(rows, idxs, cats_arr, x_by_col, cat):
    cat_mask = cats_arr[idxs] == cat
    local_idxs = idxs[cat_mask]
    out = []
    if len(local_idxs) == 0:
        return out
    for col, x_all in x_by_col.items():
        vals = x_all[local_idxs]
        for th in thresholds(vals):
            out.append((f"{col}>={th:.6g}", col, ">=", th,
                        lambda all_idxs, c=col, t=th: x_by_col[c][all_idxs] >= t))
            out.append((f"{col}<={th:.6g}", col, "<=", th,
                        lambda all_idxs, c=col, t=th: x_by_col[c][all_idxs] <= t))
    return out


def select_rule_for_cat(cat, train_idx, cats_arr, x_by_col, lam, s1_tp, s1_pred, s2_tp, s2_pred):
    base_cat_train = train_idx[cats_arr[train_idx] == cat]
    if len(base_cat_train) == 0:
        return {"cat": cat, "rule": "never", "gain": 0.0, "n_train_use": 0}

    conditions = make_single_conditions(None, train_idx, cats_arr, x_by_col, cat)

    candidates = [("never", np.zeros(len(train_idx), dtype=bool))]
    # Single-condition candidates.
    scored_singles = []
    for name, _col, _op, _th, fn in conditions:
        mask = np.zeros(len(train_idx), dtype=bool)
        cat_pos = cats_arr[train_idx] == cat
        cond = fn(train_idx)
        mask[cat_pos & cond] = True
        n_use = int(mask.sum())
        if n_use == 0 or n_use > 25:
            continue
        gain = rule_gain(mask, lam, s1_tp[train_idx], s1_pred[train_idx],
                         s2_tp[train_idx], s2_pred[train_idx])
        scored_singles.append((gain, name, fn, n_use, mask))
        candidates.append((name, mask))

    # Pairwise AND among strongest single conditions. This makes category-specific
    # hard gates expressive without exploding the search.
    scored_singles.sort(key=lambda x: x[0], reverse=True)
    top = scored_singles[:20]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            name = f"({top[i][1]}) AND ({top[j][1]})"
            cond = top[i][2](train_idx) & top[j][2](train_idx)
            mask = np.zeros(len(train_idx), dtype=bool)
            cat_pos = cats_arr[train_idx] == cat
            mask[cat_pos & cond] = True
            n_use = int(mask.sum())
            if n_use == 0 or n_use > 25:
                continue
            candidates.append((name, mask))

    best_name = "never"
    best_gain = 0.0
    best_n = 0
    for name, mask in candidates:
        gain = rule_gain(mask, lam, s1_tp[train_idx], s1_pred[train_idx],
                         s2_tp[train_idx], s2_pred[train_idx])
        # Strictly require positive gain to avoid forcing S2 in a category.
        if gain > best_gain:
            best_name = name
            best_gain = gain
            best_n = int(mask.sum())

    return {"cat": cat, "rule": best_name, "gain": round(best_gain, 3), "n_train_use": best_n}


def apply_rule(rule_name: str, cat: str, idxs, cats_arr, x_by_col):
    mask = np.zeros(len(idxs), dtype=bool)
    cat_pos = cats_arr[idxs] == cat
    if rule_name == "never":
        return mask

    def eval_atom(atom: str):
        if ">=" in atom:
            col, th = atom.split(">=")
            return x_by_col[col][idxs] >= float(th)
        if "<=" in atom:
            col, th = atom.split("<=")
            return x_by_col[col][idxs] <= float(th)
        raise ValueError(f"Bad atom: {atom}")

    if ") AND (" in rule_name:
        left, right = rule_name[1:-1].split(") AND (")
        cond = eval_atom(left) & eval_atom(right)
    else:
        cond = eval_atom(rule_name)
    mask[cat_pos & cond] = True
    return mask


def run_once():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    cats_arr = np.array([r["meta__category"] for r in rows], dtype=object)
    groups = np.array([r["meta__case_name"] for r in rows], dtype=object)
    s1_tp = np.array([as_float(r, "target__s1_tp") for r in rows], dtype=np.float64)
    s1_pred = np.array([as_float(r, "target__s1_pred_voxels") for r in rows], dtype=np.float64)
    s2_tp = np.array([as_float(r, "target__s2_raw_tp") for r in rows], dtype=np.float64)
    s2_pred = np.array([as_float(r, "target__s2_raw_pred_voxels") for r in rows], dtype=np.float64)
    gt = np.array([as_float(r, "target__gt_voxels") for r in rows], dtype=np.float64)
    x_by_col = {c: np.array([as_float(r, c) for r in rows], dtype=np.float64)
                for c in RULE_FEATURES if c in rows[0]}

    always = micro_dice(float(s1_tp.sum()), float(s1_pred.sum()), float(gt.sum()))

    all_tp = []
    all_pred = []
    all_gt = []
    fold_rules = []
    total_use = 0

    gkf = GroupKFold(n_splits=5)
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(rows)), groups=groups), start=1):
        lam = dinkelbach(s1_tp[train_idx], s1_pred[train_idx],
                         s2_tp[train_idx], s2_pred[train_idx], gt[train_idx])
        selected = []
        test_use = np.zeros(len(test_idx), dtype=bool)
        for cat in CANDIDATE_CATS:
            rule = select_rule_for_cat(cat, train_idx, cats_arr, x_by_col, lam,
                                       s1_tp, s1_pred, s2_tp, s2_pred)
            selected.append(rule)
            test_use |= apply_rule(rule["rule"], cat, test_idx, cats_arr, x_by_col)

        tp = np.where(test_use, s2_tp[test_idx], s1_tp[test_idx]).sum()
        pred = np.where(test_use, s2_pred[test_idx], s1_pred[test_idx]).sum()
        g = gt[test_idx].sum()
        all_tp.append(tp)
        all_pred.append(pred)
        all_gt.append(g)
        total_use += int(test_use.sum())
        fold_rules.append({
            "fold": fold_i,
            "lambda": round(lam, 6),
            "n_test_use_s2": int(test_use.sum()),
            "test_dice": round(micro_dice(float(tp), float(pred), float(g)), 4),
            "rules": selected,
        })

    pooled = micro_dice(float(sum(all_tp)), float(sum(all_pred)), float(sum(all_gt)))
    return {
        "always_s1": round(always, 4),
        "hard_gate_pooled_oof": round(pooled, 4),
        "n_use_s2": total_use,
        "folds": fold_rules,
    }


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    result = run_once()
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Always S1: {result['always_s1']:.4f}")
    print(f"Hard gate pooled OOF: {result['hard_gate_pooled_oof']:.4f}  n_use_s2={result['n_use_s2']}")
    for fold in result["folds"]:
        print(f"\nFold {fold['fold']} dice={fold['test_dice']:.4f} use_s2={fold['n_test_use_s2']}")
        for r in fold["rules"]:
            if r["rule"] != "never":
                print(f"  {r['cat']}: {r['rule']}  train_gain={r['gain']} n_train={r['n_train_use']}")
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
