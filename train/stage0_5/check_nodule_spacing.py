import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "stage0_workspace/stage0_v2/router"))

import nibabel as nib
from router_utils import read_jsonl

failed_findings = {
    "finding_004747": "1e", "finding_001004": "2d", "finding_005559": "2d",
    "finding_005568": "2d", "finding_001910": "2d", "finding_007426": "2d",
}

manifest_path = Path(
    r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛"
    r"\生医工大赛-demo\full_dataset_work\clean_full\manifests"
    r"\finding_mask_manifest.jsonl"
)
manifest = read_jsonl(manifest_path)

# Also check some successful ones for comparison
successful = {"finding_004748": "2d"}
failed_findings.update(successful)

lu16_spacing = (0.703125, 0.703125, 1.25)

for r in manifest:
    if r["id"] not in failed_findings:
        continue
    case_name = r["case_name"]
    cat = failed_findings[r["id"]]

    ct_paths = [
        Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛"
             r"\stage1_verification\data\images") / case_name,
        Path(r"C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛"
             r"\stage1_verification\data\images_flat") / case_name,
    ]
    ct_path = None
    for cp in ct_paths:
        if cp.exists():
            ct_path = cp
            break

    if ct_path is None:
        print(f"{r['id']} cat={cat}: CT NOT FOUND")
        continue

    nii = nib.load(str(ct_path))
    zooms = nii.header.get_zooms()[:3]
    scales = [z / t for z, t in zip(zooms, lu16_spacing)]

    print(f"\n{r['id']} cat={cat} {'FAILED' if r['id']!='finding_004748' else 'OK->hit'}")
    print(f"  CT: {ct_path.name}")
    print(f"  shape: {nii.shape}")
    print(f"  spacing: {zooms[0]:.4f} x {zooms[1]:.4f} x {zooms[2]:.4f} mm")
    print(f"  LUNA16:  0.7031 x 0.7031 x 1.2500 mm")
    print(f"  scale:   {scales[0]:.3f}x  {scales[1]:.3f}x  {scales[2]:.3f}x")
    if max(abs(s - 1) for s in scales) > 0.1:
        print(f"  *** MISMATCH! 最大偏离 {max(abs(s - 1) for s in scales):.1%}")
