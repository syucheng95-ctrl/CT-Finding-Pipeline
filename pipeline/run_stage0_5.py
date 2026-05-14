"""Stage0.5 runner: MoE proposal generator on Stage0 pre-cropped ROIs.

Calls stage0.5_workspace/proposal_generator/run_stage0_5.py (which uses
Stage0's router predictions + crop groups to generate proposals on the
already-cropped ROI images). Uses `-m proposal_generator.run_stage0_5`
because the script has relative imports (from .input_parser etc.).
"""

import argparse
import os
from pathlib import Path

from utils import load_config, resolve, run_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage0.5 proposal generator runner")
    parser.add_argument("--config", default="config.yaml", help="Path to pipeline config")
    parser.add_argument("--limit", type=int, default=0, help="Limit N findings (0=all)")
    args = parser.parse_args()

    config = load_config(args.config)
    python = config["python"]
    upload_dir = Path(config["_upload_dir"])
    out_dir = Path(resolve(config, "outputs.stage0_5"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage0 outputs → Stage0.5 inputs
    stage0_out = Path(resolve(config, "outputs.stage0"))
    predictions = str(stage0_out / "stage0_router_predictions.jsonl")
    # stage0_crop_groups.jsonl contains the merged Stage0 crop groups.
    crop_manifest = stage0_out / "stage0_crop_groups.jsonl"
    if not crop_manifest.exists():
        # Fallback: check individual case dirs for merged groups
        crop_groups_files = list(stage0_out.glob("**/stage0_crop_groups.jsonl"))
        if crop_groups_files:
            # Use the first one for now (single case) or merge
            crop_groups = str(crop_groups_files[0])
        else:
            print("[ERROR] No crop_groups.jsonl found from Stage0")
            raise SystemExit(1)
    else:
        crop_groups = str(crop_manifest)

    stage0_5_workspace = str(upload_dir / "workspace")  # parent of proposal_generator/

    # PYTHONPATH + cwd: run_stage0_5.py has relative imports and needs
    # proposal_generator as a package, plus STAGE0_V2 modules
    stage0_v2_router = str(upload_dir / "workspace" / "stage0")
    env = os.environ.copy()
    paths = [stage0_v2_router, stage0_5_workspace]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    cmd = [
        python, "-m", "proposal_generator.run_stage0_5",
        "--predictions", predictions,
        "--crop-groups", crop_groups,
        "--out-dir", str(out_dir),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    run_step(cmd, "Stage0.5: Proposal Generation",
             env=env, cwd=stage0_5_workspace)


if __name__ == "__main__":
    main()
