"""Extract per-ROI kymograph traces from every linescan unit in a .mesc file.

Usage:
    python scripts/extract_todd_linescan.py [mesc_path] [out_root]

Defaults to Todd's real line-scan file / a results dir under
~/repos/todd_project so his raw data folder is never written to. Prints
per-unit shape and trace summary stats so the numbers can be sanity-checked
against the acquisition comments before handing anything off.
"""

import sys
from pathlib import Path

import numpy as np

from mbo_utilities.roi_workflow import extract_linescan_units

DEFAULT_MESC = Path(r"C:\Users\loson\data\todd\TZ_SCE007_LINE_SCAN_SPI_20260730.mesc")
DEFAULT_OUT_ROOT = Path.home() / "repos" / "todd_project" / "results" / "linescan"


def _summarize(path: Path, arr_name: str) -> str:
    arr = np.load(path / arr_name)
    return f"shape={arr.shape} mean={arr.mean():.3f} std={arr.std():.3f} min={arr.min():.3f} max={arr.max():.3f}"


def main() -> None:
    mesc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MESC
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"mesc file: {mesc_path}")
    print(f"writing results to: {out_root}\n")

    outputs = extract_linescan_units(mesc_path, out_root=out_root)
    if not outputs:
        print("no linescan (kind='packed') units found in this file.")
        return

    for unit_key, out_dir in outputs.items():
        print(f"=== {unit_key} -> {out_dir} ===")
        print(f"  F:    {_summarize(out_dir, 'F.npy')}")
        print(f"  dfof: {_summarize(out_dir, 'dfof.npy')}")
        rois = np.load(out_dir / "stat.npy", allow_pickle=True)
        print(f"  {len(rois)} ROIs, widths={[int(r['width']) for r in rois]}")
        print()


if __name__ == "__main__":
    main()
