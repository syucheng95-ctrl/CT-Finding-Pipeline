"""③ Qwen structured extraction — batch process finding texts via Ollama."""

import json
import re
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "dolphin-llama3:latest"

SYSTEM_PROMPT = """You are a radiology report structuring assistant. Extract structured fields from the given radiology finding text.

Fields:
- lesion_type: one of [nodule, mass, GGO, consolidation, effusion, pneumothorax, fibrosis, bronchial_thickening, bronchiectasis, emphysema, atelectasis, calcification, other]
- organ: the organ mentioned (lung, pleura, liver, kidney, adrenal, pancreas, bone, lymph_node, mediastinum, other)
- location: anatomical location description in English
- stated_diameter_mm: explicit size in mm if mentioned in the text, else null
- laterality: left, right, bilateral, or null

Rules:
1. If a diameter is stated (e.g. "6 mm", "1.2 cm"), convert to mm.
2. If the text mentions multiple lesions, describe the primary one.
3. Output ONLY valid JSON, no markdown, no explanation.

Example output:
{"lesion_type": "nodule", "organ": "lung", "location": "right upper lobe posterior segment", "stated_diameter_mm": 6.0, "laterality": "right"}
"""


def _call_ollama(prompt_text, timeout=60):
    """Single Ollama call, return parsed JSON dict or None."""
    full_prompt = f"{SYSTEM_PROMPT}\n\nText: {prompt_text}\nJSON:"
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
        # Extract JSON block if wrapped
        if "```" in raw:
            raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return None


def _regex_fallback(text):
    """Extract whatever we can with regex when Qwen fails."""
    result = {
        "lesion_type": "other", "organ": "lung",
        "location": "", "stated_diameter_mm": None, "laterality": None,
    }
    # Diameter
    m = re.search(r"(\d+\.?\d*)\s*(mm|cm)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if m.group(2).lower() == "cm":
            val *= 10
        result["stated_diameter_mm"] = val
    # Laterality
    if re.search(r"\bleft\b", text, re.IGNORECASE):
        result["laterality"] = "left"
    if re.search(r"\bright\b", text, re.IGNORECASE):
        result["laterality"] = "right" if result["laterality"] is None else "bilateral"
    if re.search(r"\bbilateral\b|\bboth\s+lungs?\b", text, re.IGNORECASE):
        result["laterality"] = "bilateral"
    # Lesion type keywords
    kw_map = {
        "nodule": [r"\bnodul", r"\bmicronodul"],
        "GGO": [r"\bground.glass\b", r"\bggo\b", r"\bground glass\b"],
        "consolidation": [r"\bconsolidation\b"],
        "effusion": [r"\beffusion\b", r"\bpleural fluid\b"],
        "pneumothorax": [r"\bpneumothorax\b"],
        "atelectasis": [r"\batelectasis\b"],
        "fibrosis": [r"\bfibrosis\b", r"\bfibrotic\b"],
        "emphysema": [r"\bemphysema\b"],
        "bronchial_thickening": [r"\bbronchial wall thick"],
        "bronchiectasis": [r"\bbronchiectasis\b"],
    }
    for lt, pats in kw_map.items():
        if any(re.search(p, text, re.IGNORECASE) for p in pats):
            result["lesion_type"] = lt
            break
    return result


def _infer_metric_group(record):
    """Map structured fields to metric_group."""
    lt = record.get("lesion_type", "other")
    organ = record.get("organ", "lung")
    if organ != "lung" and organ != "pleura":
        return "lung_opacity"  # default for non-thoracic
    lung_types = {"nodule", "mass", "GGO", "consolidation", "atelectasis",
                  "emphysema", "calcification", "other"}
    airway_types = {"bronchial_thickening", "bronchiectasis"}
    pleural_types = {"effusion", "pneumothorax", "fibrosis"}
    # Pleural types only count if organ is pleura (not lung)
    if lt in pleural_types and organ == "pleura":
        return "pleural"
    if lt in airway_types:
        return "airway_change"
    return "lung_opacity"


# ═══════════════════════════════════════════════════════════════
# Normalization: map free-form model outputs to controlled vocab
# ═══════════════════════════════════════════════════════════════

_LESION_TYPE_MAP = {
    "bronchial_wall_thickening": "bronchial_thickening",
    "reticulonodular_density": "fibrosis",
    "bulla": "emphysema",
    "calcific_nodule": "calcification",
    "calcified_nodule": "calcification",
    "secretion": "other",
    "bronchiolitis": "bronchial_thickening",
    "pneumonic_infiltration": "consolidation",
    "pulmonary_nodule": "nodule",
    "ground_glass_opacity": "GGO",
}
_ORGAN_MAP = {
    "trachea": "other", "bronchus": "other",
    "trachea_and_bronchi": "other", "trachea_and_bronchus": "other",
    "chest_wall": "other", "lungs": "lung",
}
_VALID_LATERALITY = {"left", "right", "bilateral"}


def normalize_record(record):
    """Clean up a single record's fields to standard vocabulary."""
    lt = record.get("lesion_type")
    if lt is None or lt == "None":
        record["lesion_type"] = "other"
    elif lt in _LESION_TYPE_MAP:
        record["lesion_type"] = _LESION_TYPE_MAP[lt]

    org = record.get("organ")
    if org is None or org == "None":
        record["organ"] = "lung"
    elif org in _ORGAN_MAP:
        record["organ"] = _ORGAN_MAP[org]

    lat = record.get("laterality")
    if lat not in _VALID_LATERALITY:
        record["laterality"] = None

    # Effusion / pneumothorax are always pleural
    if record.get("lesion_type") in ("effusion", "pneumothorax"):
        record["organ"] = "pleura"

    # Fix stated_diameter_mm: collapse list → max, string → float, reject negatives
    dia = record.get("stated_diameter_mm")
    if isinstance(dia, list):
        dia = max((d for d in dia if isinstance(d, (int, float))), default=None)
    if isinstance(dia, str):
        try:
            dia = float(dia)
        except ValueError:
            dia = None
    if dia is not None and dia <= 0:
        dia = None
    # Cap unreasonable diameters > 200mm (likely cm→mm error)
    if dia is not None and dia > 200:
        dia = None
    record["stated_diameter_mm"] = dia

    # Re-infer metric_group in case lesion_type changed
    record["metric_group"] = _infer_metric_group(record)
    return record


def process_findings(finding_texts, finding_ids=None, checkpoint_path=None):
    """Batch process findings, returning list of structured dicts.

    Args:
        finding_texts: list of raw finding strings.
        finding_ids: optional list of IDs.
        checkpoint_path: optional path to save/resume progress.

    Returns:
        list of dicts with keys: finding_id, lesion_type, organ, location,
        stated_diameter_mm, laterality, metric_group, _source (qwen/regex).
    """
    if finding_ids is None:
        finding_ids = [f"finding_{i:06d}" for i in range(len(finding_texts))]

    # Load checkpoint if exists
    done = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["finding_id"]] = r

    results = []
    n_total = len(finding_texts)
    for idx, (fid, text) in enumerate(zip(finding_ids, finding_texts)):
        if fid in done:
            results.append(done[fid])
            continue

        parsed = _call_ollama(text)
        # Qwen occasionally returns a JSON list instead of object
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if isinstance(parsed, dict):
            parsed["_source"] = "qwen"
        else:
            parsed = _regex_fallback(text)
            parsed["_source"] = "regex"

        parsed.setdefault("finding_id", fid)
        parsed.setdefault("lesion_type", "other")
        parsed.setdefault("organ", "lung")
        parsed.setdefault("location", "")
        parsed.setdefault("stated_diameter_mm", None)
        parsed.setdefault("laterality", None)
        parsed["metric_group"] = _infer_metric_group(parsed)

        results.append(parsed)

        if checkpoint_path and (idx + 1) % 100 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Checkpoint: {idx + 1}/{n_total} ({100*(idx+1)/n_total:.0f}%)")

        if (idx + 1) % 20 == 0:
            lt = parsed.get('lesion_type') or 'other'
            src = parsed.get('_source', '?')
            print(f"  [{idx + 1}/{n_total}] {src:5s}  "
                  f"{lt:20s}  {text[:60]}...")

    # Final save
    if checkpoint_path:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(results)} records to {checkpoint_path}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from stage2_5.classifier.config import ROI_MANIFEST

    print("Loading finding texts...")
    with open(ROI_MANIFEST) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # Deduplicate by prompt text
    seen = {}
    for r in rows:
        prompt = r["prompt"].strip()
        if prompt not in seen:
            seen[prompt] = r.get("parent_id", "")  # finding_id
    texts = list(seen)
    fids = [seen[t] for t in texts]
    print(f"  {len(rows)} ROIs → {len(texts)} unique prompts")

    out_path = Path(__file__).resolve().parent.parent / "outputs" / "structured_findings.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing with {MODEL}...")
    process_findings(texts, fids, checkpoint_path=out_path)
    print("Done.")
