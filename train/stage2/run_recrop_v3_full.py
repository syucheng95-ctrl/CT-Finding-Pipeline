from pathlib import Path
import os

import recrop_v3_rois_stream


ROOT = Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛")
DEMO = ROOT / "生医工大赛-demo"
WORK = DEMO / "full_dataset_work"
OLD_ROI = WORK / "stage2_roi_50case"
OUT = ROOT / "stage2_workspace" / "outputs" / "stage2_roi_v3"


def main():
    argv = [
        "recrop_v3_rois_stream.py",
        "--manifest", str(OLD_ROI / "stage2_train_manifest.jsonl"),
        "--roi-manifest", str(OLD_ROI / "roi_manifest.jsonl"),
        "--old-roi-root", str(OLD_ROI),
        "--out-root", str(OUT),
        "--out-manifest", str(OUT / "manifest_v3.jsonl"),
        "--temp-root", str(ROOT / "stage2_workspace" / "outputs" / "_temp_clean_full_v3"),
        "--groups", "large", "xlarge",
        "--ratio", "0.25",
        "--min-margin-hwd", "16", "16", "16",
        "--copy-unchanged",
        "--resume",
        "--progress-json", str(OUT / "recrop_v3_stream_progress.json"),
        "--flush-every-cases", "5",
    ]
    limit = os.environ.get("V3_RECROP_LIMIT")
    if limit:
        argv.extend(["--limit", limit])
    import sys

    sys.argv = argv
    recrop_v3_rois_stream.main()


if __name__ == "__main__":
    main()
