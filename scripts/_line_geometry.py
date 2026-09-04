"""Reconstruct where a Femtonics line-scan's drawn line sits in a Z-stack.

Neither `lab4/convert/aod/aod.py` nor `mbo_utilities/arrays/mesc.py` keeps
what's needed for this: both reduce a linescan ROI's guideline down to a
centroid + rotation quaternion for metadata (`mesc_centroids`,
`mesc_rotations`) - enough to know roughly where a ROI is, not its actual
start/end line segment. See `.claude/plans/flickering-roaming-squid.md` for
how this was found (real HDF5 attrs on a real file, cross-checked against
independently-known pixel widths - not documented anywhere existing).

Attrs read directly via h5py (bypassing MescArray.metadata, which doesn't
carry any of these):

- A linescan unit's `CoordinateMapJSON["maps"][0]["driftEndPoints"]`: one
  `[[x0,x1],[y0,y1],[z0,z1]]` per ROI, in physical microns. Each ROI is
  scanned at ONE depth (z0 == z1 on every real ROI checked) - AOD systems
  steer the beam to a different (x,y,z) target per ROI within one frame
  period, so different ROIs in the same linescan unit genuinely sit at
  different depths. A ROI's line only belongs on the Z-stack slice at that
  depth, not on every slice - see `roi_slice_indices`.
- A Z-stack unit's `ReferenceViewportJSON["viewports"][0]`: the FOV's
  physical placement (`geomTransTransl`, `width`, `height`) - XY corner AND
  the Z-stack's own Z origin, in the SAME micron coordinate frame the
  linescan's driftEndPoints are in.
- A Z-stack unit's `MinZ`/`MaxZ`/`ZDim` attrs: the stack's depth range and
  slice count, assumed uniformly stepped (`ZModeDebugString` on the units
  checked was `"slow"`, a real raster Z-stack, not an arbitrarily-placed-ROI
  AOD scan - no per-slice Z array was found in `AOSettingsJSON` to say
  otherwise).

**Z-origin correction (found by cross-checking against Todd's real data,
not documented anywhere):** the Z-stack's Z origin is
`ReferenceViewportJSON.geomTransTransl[2]`, NOT the unit's
`LabelingOriginTransl` attr (a different, unrelated translation - using it
put a real linescan's depth range entirely outside every candidate
Z-stack's computed range, which is what caused lines to land on a FOV with
no visible dendrite: XY was already using `geomTransTransl` correctly, but
the *pairing check* that would have caught a wrong Z-stack choice was using
the wrong Z origin, so nothing flagged it as be being off).

Every function here returns `None` when its underlying attr is missing
(older MESc version, non-AOD acquisition) so a caller can skip the overlay
cleanly rather than crash.
"""

from __future__ import annotations

import json

import h5py
import numpy as np


def linescan_endpoints_um(mesc_path, unit_key: str) -> list[np.ndarray] | None:
    """One (3, 2) `[[x0,x1],[y0,y1],[z0,z1]]` array per ROI, in microns.

    `None` if this unit has no `CoordinateMapJSON` or no `driftEndPoints`
    inside it (present on real linescan units, not guaranteed on others).
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None or "CoordinateMapJSON" not in unit.attrs:
            return None
        raw = unit.attrs["CoordinateMapJSON"]
        if not raw:
            return None
        doc = json.loads(raw)
        maps = doc.get("maps") or []
        if not maps or "driftEndPoints" not in maps[0]:
            return None
        return [np.asarray(seg, dtype=float) for seg in maps[0]["driftEndPoints"]]


def viewport_geometry(mesc_path, unit_key: str) -> dict | None:
    """`{"transl": (x, y, z), "width": um, "height": um}` for a unit's FOV.

    `None` if this unit has no `ReferenceViewportJSON`.
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None or "ReferenceViewportJSON" not in unit.attrs:
            return None
        raw = unit.attrs["ReferenceViewportJSON"]
        if not raw:
            return None
        doc = json.loads(raw)
        viewports = doc.get("viewports") or []
        if not viewports:
            return None
        vp = viewports[0]
        return {
            "transl": tuple(float(v) for v in vp["geomTransTransl"]),
            "width": float(vp["width"]),
            "height": float(vp["height"]),
        }


def zstack_depth_info(mesc_path, unit_key: str) -> dict | None:
    """`{"transl_z", "min_z", "max_z", "zdim"}` for a Z-stack unit's depth axis.

    `None` if this unit is missing `ReferenceViewportJSON`, `MinZ`, `MaxZ`,
    or `ZDim`. `transl_z` is `geomTransTransl[2]` (see module docstring for
    why this and not `LabelingOriginTransl`).
    """
    vp = viewport_geometry(mesc_path, unit_key)
    if vp is None:
        return None
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None or not {"MinZ", "MaxZ", "ZDim"} <= set(unit.attrs.keys()):
            return None
        return {
            "transl_z": vp["transl"][2],
            "min_z": float(unit.attrs["MinZ"]),
            "max_z": float(unit.attrs["MaxZ"]),
            "zdim": int(unit.attrs["ZDim"]),
        }


def roi_slice_indices(lines_um: list[np.ndarray], depth: dict) -> np.ndarray:
    """0-based Z-stack slice index per ROI, from each line's mean z.

    Rounds to the nearest slice - the Z slider moves in integer steps, so
    a ROI's line should appear on exactly one slice, not fade across two.
    Indices are clipped into `[0, zdim - 1]`; a ROI scanned outside the
    stack's captured depth range still gets a (clamped) slice rather than
    silently vanishing, since that's more useful for spotting a real
    depth-range mismatch than a ROI that just never shows up.
    """
    zdim = depth["zdim"]
    step = (depth["max_z"] - depth["min_z"]) / max(zdim - 1, 1)
    z_means = np.array([float(seg[2].mean()) for seg in lines_um])
    idx = np.round((z_means - depth["transl_z"] - depth["min_z"]) / step).astype(int)
    return np.clip(idx, 0, zdim - 1)


def um_to_pixels(
    points_xy_um: np.ndarray,
    viewport: dict,
    ny: int,
    nx: int,
    flip_y: bool = False,
) -> np.ndarray:
    """`(N, 2)` physical `[x, y]` microns -> `(N, 2)` pixel `[col, row]`.

    `geomTransTransl` is treated as the FOV's lower corner (verified against
    real data - see module docstring / the plan doc), not its center.
    `flip_y=True` mirrors row about the image's vertical center - the
    Y-orientation convention is an open question across this codebase
    (Femtonics.md Sec 5 #1); this is the one-line switch to try if lines land
    upside down relative to the real dendrite.
    """
    tx, ty, _tz = viewport["transl"]
    px_w = viewport["width"] / nx
    px_h = viewport["height"] / ny
    pts = np.asarray(points_xy_um, dtype=float)
    col = (pts[:, 0] - tx) / px_w
    row = (pts[:, 1] - ty) / px_h
    if flip_y:
        row = ny - row
    return np.stack([col, row], axis=1)
