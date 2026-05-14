"""End-to-end pipeline orchestrator.

Chains all stages sequentially:
  Stage0      → Router + Anatomy gate + CT cropping
  Stage0.5    → MoE proposal generation
  Stage1      → VoxTell baseline verification
  Stage1_metrics → S1 metrics evaluation
  Stage2_roi  → STU-Net ROI generation
  Stage2_eval → STU-Net finding-level eval (with overlap + confidence features)
  gate_table  → Build gate training table
  gate_fit    → Fit final learned gate
  gate_apply  → Apply learned gate, output final decisions

Usage:
  python run_full.py                          # default: default manifest
  python run_full.py --config config.yaml     # custom config
  python run_full.py --manifest path/to.jsonl # custom manifest
  python run_full.py --stages 0,0.5           # run only specific stages
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Ensure upload/pipeline/ is found before system packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gate"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "external"))

from utils import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline orchestrator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest", help="Manifest JSONL (default: config manifests.default)")
    parser.add_argument(
        # 默认推理模式，直接用已有 gate_model.joblib
        # 如需重训 gate：加 gate_fit 阶段
        #   python run_full.py --stages 0,...,gate_table,gate_fit,gate_apply
        "--stages", default="0,0.5,1,1_metrics,2_roi,2_eval,gate_table,gate_apply",
        help="Comma-separated stages to run",
    )
    parser.add_argument("--limit-cases", type=int, default=0, help="Limit N CT cases")
    parser.add_argument("--no-clean", action="store_true", help="Keep intermediate NIfTI files (default: auto-clean)")
    args = parser.parse_args()

    config = load_config(args.config)
    python = config["python"]
    upload_dir = Path(config["_upload_dir"])
    config_path = Path(config["_upload_dir"]) / args.config if not Path(args.config).is_absolute() else Path(args.config)
    stages = [s.strip() for s in args.stages.split(",")]

    manifest = args.manifest or resolve(config, "manifests.default")

    total_t0 = time.time()

    stage_scripts = {
        "0":           Path(config["_upload_dir"]) / "pipeline" / "run_stage0.py",
        "0.5":         Path(config["_upload_dir"]) / "pipeline" / "run_stage0_5.py",
        "1":           Path(config["_upload_dir"]) / "pipeline" / "run_stage1_verify.py",
        "1_metrics":   Path(config["_upload_dir"]) / "pipeline" / "evaluate_stage1_metrics.py",
        "2_roi":       Path(config["_upload_dir"]) / "pipeline" / "run_stage2_make_rois.py",
        "2_eval":      Path(config["_upload_dir"]) / "pipeline" / "run_stage2_eval_finding_level.py",
        "gate_table":  Path(config["_upload_dir"]) / "pipeline" / "build_gate_training_table.py",
        "gate_fit":    Path(config["_upload_dir"]) / "gate" / "train_gate.py",
        "gate_apply":  Path(config["_upload_dir"]) / "gate" / "apply_gate.py",
    }

    # Gate stages don't take --config, they use default paths
    GATE_STAGES = {"gate_table", "gate_fit", "gate_apply"}

    for stage in stages:
        if stage not in stage_scripts:
            print(f"Unknown stage: {stage}. Choose from: {sorted(stage_scripts.keys())}")
            sys.exit(1)

        script = stage_scripts[stage]

        if stage in GATE_STAGES:
            cmd = [python, str(script)]
        else:
            cmd = [
                python, str(script),
                "--config", str(config_path),
            ]

        # Per-stage extra args
        if stage == "0":
            cmd += ["--manifest", manifest]
            if args.limit_cases:
                cmd += ["--limit-cases", str(args.limit_cases)]
        elif stage == "0.5":
            if args.limit_cases:
                cmd += ["--limit", str(args.limit_cases)]
        elif stage == "1":
            if args.limit_cases:
                cmd += ["--limit-findings", str(args.limit_cases)]
        elif stage == "1_metrics":
            pass
        elif stage == "2_roi":
            cmd += ["--manifest", manifest]
            if args.limit_cases:
                cmd += ["--limit-cases", str(args.limit_cases)]
        elif stage == "2_eval":
            pass
        elif stage == "gate_fit":
            cmd += ["--fit-final"]
        elif stage in GATE_STAGES:
            pass

        print(f"\n{'#' * 60}")
        print(f"# STAGE {stage}")
        print(f"{'#' * 60}")
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\nPipeline FAILED at Stage {stage} (exit {result.returncode})")
            sys.exit(result.returncode)

        print(f"\nStage {stage} completed in {elapsed:.0f}s")

        # ── Auto-clean intermediate NIfTI (keep JSON/JSONL/CSV + final preds) ──
        if not args.no_clean:
            outputs_root = upload_dir / "outputs"
            import shutil

            cleanup_map = {
                "1": [outputs_root / "stage0"],            # Stage1 has consumed crop NIfTI
                "2_roi": [outputs_root / "stage1" / "masks"],  # coarse masks not needed after ROI gen
                "gate_table": [
                    outputs_root / "stage0",               # final cleanup
                    outputs_root / "stage0_5",
                    outputs_root / "stage1",
                    outputs_root / "stage2" / "roi_images",
                    outputs_root / "stage2" / "roi_masks",
                ],
            }

            for target_dir in cleanup_map.get(stage, []):
                if target_dir.exists():
                    # Only delete subdirectories (NIfTI), keep JSON/JSONL/CSV at root
                    if target_dir.is_dir():
                        for item in list(target_dir.iterdir()):
                            if item.is_dir():
                                try:
                                    shutil.rmtree(item)
                                    print(f"  [clean] Removed: {item.relative_to(upload_dir)}")
                                except Exception as e:
                                    print(f"  [clean] Failed to remove {item.relative_to(upload_dir)}: {e}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete! All stages passed in {total_elapsed:.0f}s")
    print(f"{'=' * 60}")

    # Collect summaries
    summaries = {}
    stage_dirs = {
        "0": "stage0", "0.5": "stage0_5", "1": "stage1",
        "1_metrics": "stage1", "2_roi": "stage2", "2_eval": "stage2",
        "gate_table": "gate_training", "gate_fit": "gate_training", "gate_apply": "final",
    }
    for s in stages:
        name = stage_dirs.get(s, s)
        if name == "final":
            summary_path = upload_dir / "outputs" / "final" / "final_metrics.json"
        elif name == "gate_training":
            if s == "gate_fit":
                summary_path = upload_dir / "gate" / "outputs" / "gate_metadata.json"
                if summary_path.exists():
                    with open(summary_path, encoding="utf-8") as f:
                        summaries[f"stage_{s}"] = json.load(f)
                continue
            summary_path = upload_dir / "gate" / "outputs" / "gate_training_table.csv"
            if summary_path.exists():
                summaries[f"stage_{s}"] = {"table_path": str(summary_path), "status": "ok"}
            continue
        else:
            summary_path = Path(resolve(config, f"outputs.{name}")) / "summary.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                summaries[f"stage_{s}"] = json.load(f)

    summary_file = upload_dir / "outputs" / "pipeline_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline_summary = {
        "manifest": manifest,
        "stages_run": stages,
        "total_time_s": round(total_elapsed, 1),
        "stage_summaries": summaries,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()
