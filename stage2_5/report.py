"""④ Report generation — fuse image measurements + structured text fields.

Produces a standardized JSON report per lesion, including comparison
between radiologist-stated measurements and CT-quantified values.
"""

import json
from pathlib import Path


def _compare_diameters(stated_mm, measured_mm):
    """Return a human-readable discrepancy note, or None if no comparison possible."""
    if stated_mm is None or measured_mm is None:
        return None
    diff = measured_mm - stated_mm
    if abs(diff) < 0.5:
        return f"影像实测 {measured_mm:.1f}mm，与报告描述 {stated_mm:.0f}mm 一致"
    direction = "大于" if diff > 0 else "小于"
    return (f"影像实测 {measured_mm:.1f}mm，{direction}报告描述的 "
            f"{stated_mm:.0f}mm（差异 {abs(diff):.1f}mm）")


def _solid_note(metrics):
    """Generate clinical interpretation of solid/GGO composition."""
    solid = metrics.get("solid_component_ratio", 0)
    ggo = metrics.get("ggo_ratio", 0)
    parts = []
    if solid > 0.5:
        parts.append(f"实性成分占 {solid*100:.0f}%，提示恶性风险较高")
    elif solid > 0.1:
        parts.append(f"实性成分占 {solid*100:.0f}%，部分实性结节")
    if ggo > 0.5:
        parts.append(f"磨玻璃成分占 {ggo*100:.0f}%")
    if not parts:
        return None
    return "；".join(parts)


def _spiculation_note(spic_idx):
    if spic_idx is None:
        return None
    if spic_idx > 1.3:
        return f"毛刺指数 {spic_idx:.2f}（>1.3），提示恶性可能"
    return f"毛刺指数 {spic_idx:.2f}，轮廓较光滑"


def _cavity_note(cav_frac):
    if cav_frac is None or cav_frac < 0.01:
        return None
    return f"内部空洞/坏死占比 {cav_frac*100:.0f}%"


def _wall_thickness_note(wt_ratio):
    if wt_ratio is None:
        return None
    if wt_ratio > 0.5:
        return f"支气管壁厚比 {wt_ratio:.2f}，提示管壁增厚"
    return None


def _effusion_note(vol_ml):
    if vol_ml is None or vol_ml < 1:
        return None
    if vol_ml > 500:
        return f"大量胸腔积液（{vol_ml:.0f}ml）"
    if vol_ml > 100:
        return f"中等量胸腔积液（{vol_ml:.0f}ml）"
    return f"少量胸腔积液（{vol_ml:.0f}ml）"


def generate_report(image_measurements, clinical_context, finding_id=None,
                    metric_group=None, category=None):
    """Generate a standardized structured report for a single lesion.

    Args:
        image_measurements: dict from ``compute_all_metrics()`` or similar,
            containing ``basic`` and optionally group-specific fields.
        clinical_context: dict from structured findings (Qwen output),
            with keys: lesion_type, organ, location, stated_diameter_mm,
            laterality.
        finding_id: optional ID string.
        metric_group: optional override (otherwise inferred from context).
        category: optional 14-class label from classifier.

    Returns:
        dict ready for JSON serialization.
    """
    basic = image_measurements.get("basic", {})
    ctx = clinical_context

    group = metric_group or image_measurements.get("metric_group",
                                                     ctx.get("metric_group", "lung_opacity"))
    stated_diam = ctx.get("stated_diameter_mm")
    measured_diam = basic.get("effective_diameter_mm") or basic.get("max_diameter_mm")

    report = {
        "finding_id": finding_id or ctx.get("finding_id", ""),
        "metric_group": group,
        "image_measurements": {
            "volume_ml": basic.get("volume_ml"),
            "max_diameter_mm": basic.get("max_diameter_mm"),
            "effective_diameter_mm": basic.get("effective_diameter_mm"),
            "mean_hu": basic.get("mean_hu"),
            "sphericity": basic.get("sphericity"),
        },
        "clinical_context": {
            "lesion_type": ctx.get("lesion_type"),
            "organ": ctx.get("organ"),
            "location": ctx.get("location"),
            "stated_diameter_mm": stated_diam,
            "laterality": ctx.get("laterality"),
        },
        "comparison": {
            "diameter_discrepancy": _compare_diameters(stated_diam, measured_diam),
        },
        "clinical_notes": [],
    }

    if category:
        report["category"] = category

    # Group-specific fields and notes
    if group == "lung_opacity":
        lo = image_measurements.get("lung_opacity", {})
        report["image_measurements"]["solid_component_ratio"] = lo.get("solid_component_ratio")
        report["image_measurements"]["ggo_ratio"] = lo.get("ggo_ratio")
        report["image_measurements"]["spiculation_index"] = lo.get("spiculation_index")
        report["image_measurements"]["pleural_distance_mm"] = lo.get("pleural_distance_mm")
        report["image_measurements"]["cavity_fraction"] = lo.get("cavity_fraction")

        for note in [_solid_note(lo), _spiculation_note(lo.get("spiculation_index")),
                     _cavity_note(lo.get("cavity_fraction"))]:
            if note:
                report["clinical_notes"].append(note)

    elif group == "airway_change":
        ac = image_measurements.get("airway_change", {})
        report["image_measurements"]["wall_thickness_ratio"] = ac.get("wall_thickness_ratio")
        report["image_measurements"]["lumen_dilation_index"] = ac.get("lumen_dilation_index")

        wt_note = _wall_thickness_note(ac.get("wall_thickness_ratio"))
        if wt_note:
            report["clinical_notes"].append(wt_note)
        ld = ac.get("lumen_dilation_index")
        if ld is not None and ld > 0.7:
            report["clinical_notes"].append(f"管径扩张指数 {ld:.2f}（>0.7），提示支气管扩张")

    elif group == "pleural":
        pl = image_measurements.get("pleural", {})
        report["image_measurements"]["effusion_volume_ml"] = pl.get("effusion_volume_ml")
        report["image_measurements"]["pleural_thickness_mm"] = pl.get("pleural_thickness_mm")
        report["image_measurements"]["pneumothorax_fraction"] = pl.get("pneumothorax_fraction")

        for note in [_effusion_note(pl.get("effusion_volume_ml"))]:
            if note:
                report["clinical_notes"].append(note)

    # Remove empty clinical_notes if nothing was added
    if not report["clinical_notes"]:
        del report["clinical_notes"]

    # Remove None from comparison
    if report["comparison"]["diameter_discrepancy"] is None:
        del report["comparison"]["diameter_discrepancy"]

    return report


def batch_generate_reports(metrics_list, context_map, categories=None):
    """Generate reports for a batch of ROIs.

    Args:
        metrics_list: list of (roi_id, metrics_dict) tuples.
        context_map: dict mapping finding_id → clinical_context dict.
        categories: optional dict mapping roi_id → category string.

    Returns:
        list of report dicts.
    """
    reports = []
    for roi_id, metrics in metrics_list:
        finding_id = metrics.get("finding_id", roi_id)
        ctx = context_map.get(finding_id, {})
        cat = categories.get(roi_id) if categories else None
        report = generate_report(
            metrics, ctx,
            finding_id=finding_id,
            metric_group=metrics.get("metric_group"),
            category=cat,
        )
        reports.append(report)
    return reports


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Load a few structured findings
    findings_path = Path(__file__).resolve().parent.parent / "outputs" / "structured_findings.jsonl"
    with open(findings_path, encoding="utf-8") as f:
        findings = [json.loads(l) for l in f if l.strip()]

    # Pick one per group
    import nibabel as nib
    import numpy as np
    from stage2_5.classifier.config import ROI_MANIFEST
    from stage2_5.metrics import compute_all_metrics

    with open(ROI_MANIFEST) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    print("Smoke test: generate one report per metric group\n")

    roi_root = Path(ROI_MANIFEST).parent
    for group in ("lung_opacity", "airway_change", "pleural"):
        # Find a ROI whose finding matches this group
        ctx = next((f for f in findings if f["metric_group"] == group), None)
        if ctx is None:
            continue
        roi = next((r for r in rows if r.get("parent_id") == ctx["finding_id"]), None)
        if roi is None:
            continue

        img = nib.load(str(roi_root / roi["roi_image"]))
        mask = nib.load(str(roi_root / roi["roi_mask"]))
        ct = np.asanyarray(img.dataobj).astype(np.float32).transpose(2, 0, 1)
        m = np.asanyarray(mask.dataobj).astype(np.uint8).transpose(2, 0, 1)
        spacing = img.header.get_zooms()

        metrics = compute_all_metrics(ct, m, spacing, roi["category"])
        report = generate_report(metrics, ctx, finding_id=ctx["finding_id"],
                                 category=roi["category"])
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print()
