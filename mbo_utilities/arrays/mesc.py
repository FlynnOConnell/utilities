"""
Femtonics MESc (.mesc) reader.

A ``.mesc`` file is an HDF5 container. It holds one or more acquisition
sessions (``MSession_N``), and each session holds one or more measurement
units (``MUnit_M``). A **MUnit is one scan** -- one thing the operator ran at
the scope: a z-stack, a ribbon time series, a single snapshot. Detector
channels live inside the unit as ``Channel_0``, ``Channel_1``, ... and every
unit carries its own attributes (``MethodType``, ``TStepInMs``, the scan
protocol JSON, timing curves).

One file therefore holds many unrelated recordings. `MescArray` opens exactly
one unit, chosen by ``session``/``munit``/``unit``; `list_mesc_units` reports
what a file contains so a caller (CLI ``--unit``, the GUI picker) can choose.

Layout
------
``MethodType`` says what axis 0 of ``Channel_N`` means, and how the scanned
sub-regions are packed into the raw page:

===========================  ==========================  =====================
MethodType                   raw ``Channel_N``           unpacking
===========================  ==========================  =====================
1  timeseries                ``(T, Y, X)``               none
2  zstack                    ``(Z, Y, X)``               none, axis 0 is depth
6  linescan                  ``(1, T*n_lines, X)``       frames packed into Y
7  multiline                 ``(1, T*n_lines, X)``       frames packed into Y
8  chessboard                ``(T, Y, R*X)``             ROIs tiled along X
9  ribbon transverse         ``(T, Y, X)``               ROI boxes on the page
10 ribbon longitudinal       ``(T, Y, X)``               ROI boxes, or tiled
===========================  ==========================  =====================

The AOD ROI index ``R`` is not a depth series: the ROIs are arbitrarily placed
patches in 3D, not evenly spaced and not necessarily monotonic in z. But
structurally ``R`` fills the slot ``Z`` fills -- "which spatial sub-volume,
sampled quasi-simultaneously within one frame period" -- so it is presented on
the canonical ``Z`` axis, with the real geometry kept in metadata
(``mesc_centroids``, ``mesc_rotations``) and ``mesc_z_axis_meaning`` set to
``"roi_index"`` so downstream code never has to guess. ``dz`` stays ``None``
for AOD multi-ROI: there is no uniform step and writing one would be a lie.

`MescArray` reports the canonical mbo 5D shape ``(T, C, Z, Y, X)``. Note this
is the mbo axis order, not the ``(T, Z, C, Y, X)`` used by ``lab4.convert``'s
H5 output contract -- ``C`` is axis 1 for every array `imread` returns.

Unpacking logic is ported from ``lab4/convert/aod/aod.py``, made lazy: a read
touches only the requested frames instead of materialising whole channels.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.analysis.phasecorr import _apply_offset, bidir_phasecorr
from mbo_utilities.arrays._base import (
    ReductionMixin,
    Shape5DMixin,
    _imwrite_base,
    _normalize_key,
)
from mbo_utilities.arrays.features import (
    PhaseCorrectionFeature,
    PhaseCorrectionMixin,
    RoiFeatureMixin,
)
from mbo_utilities.arrays.features._slicing import listify_index
from mbo_utilities.lazy_array import register_array_class
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = log.get("arrays.mesc")

_MESC_INFO = PipelineInfo(
    name="mesc",
    description="Femtonics MESc acquisitions",
    input_patterns=["**/*.mesc"],
    output_patterns=[],
    input_extensions=["mesc"],
    output_extensions=[],
    marker_files=[],
    category="reader",
)
register_pipeline(_MESC_INFO)


MODALITY_NAMES: dict[int, str] = {
    1: "timeseries",
    2: "zstack",
    6: "linescan",
    7: "multiline",
    8: "chessboard",
    9: "ribbon_transverse",
    10: "ribbon_longitudinal",
    11: "multicube",
}

# Modalities whose ROIs are unpacked by cropping boxes off the raw page.
_BOX_MODALITIES = frozenset({9, 10})
# Modalities whose ROIs are tiled contiguously along X of the raw page.
_TILED_MODALITIES = frozenset({8})
# Modalities whose temporal frames are packed into the Y axis of a 1-frame page.
_PACKED_MODALITIES = frozenset({6, 7})

# Y is flipped on read for these, matching lab4.convert. The flip corrects
# MEScan's save orientation; whether it also belongs on linescan/multiline is
# unsettled upstream (see `flip_y` in the class docstring), so the default
# reproduces the reference converter rather than guessing.
_FLIP_Y_BY_MODALITY: dict[int, bool] = {8: True, 9: True, 10: True}

# Timing curves worth keeping; the rest are scanner internals.
CURVE_NAMES = frozenset(
    {
        "PatternSeq_AO1",
        "DichroSw_AO1",
        "Amplitude_AO1",
        "RTMC X correction (total)",
        "RTMC Y correction (total)",
        "RTMC Z correction (total)",
        "disUG",
        "disUR",
        "DiI1",
        "DiI2",
    }
)

SYNC_KEY_DEFAULT = "DiI2"
SYNC_EDGE_DEFAULT = "falling"

# dichroic light-path codes in the DichroSw_AO1 curve
_GREEN_LP = 1
_RED_LP = 2

# lab4.convert stamps acquisition times in US Eastern standard offset. Kept
# explicit (and echoed in metadata as `start_time_tz`) rather than implied.
_ACQ_TZ = timezone(timedelta(hours=-5))

# attrs holding large embedded JSON; parsed products are exposed instead so
# the metadata panel doesn't have to render a multi-megabyte blob.
_JSON_ATTRS = (
    "MultiROIProtocolJSON",
    "CoordinateMapJSON",
    "BreakViewJSON",
    "DeviceJSON",
)


def _unit_sort_key(name: str) -> tuple[int, str]:
    """Sort ``MUnit_10`` after ``MUnit_9`` instead of before it."""
    tail = name.rsplit("_", 1)[-1]
    return (int(tail), name) if tail.isdigit() else (1 << 30, name)


def _attr(group, name, default=None):
    """Read one HDF5 attribute, decoding bytes, returning `default` if absent."""
    value = group.attrs.get(name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.dtype.kind in "SU":
        return "".join(v.decode() if isinstance(v, bytes) else str(v) for v in value)
    return value


def _json_attr(group, name):
    """Parse a JSON-valued attribute, or return None if missing/malformed."""
    raw = _attr(group, name)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as e:
        logger.debug(f"{name} on {group.name} is not valid JSON: {e}")
        return None


def _scan_pattern(unit) -> dict | None:
    """The imaging scan pattern this unit was acquired with, if declared."""
    protocol = _json_attr(unit, "MultiROIProtocolJSON")
    if not protocol:
        return None
    try:
        idx = protocol["protocol"]["scanners"]["mainPatternIndex"] - 1  # MATLAB 1-based
        return protocol["scanPatterns"]["patterns"][idx]
    except (KeyError, IndexError, TypeError) as e:
        logger.debug(f"no main scan pattern in {unit.name}: {e}")
        return None


def _boxes_from(pixel_rois) -> list[dict]:
    """Convert MESc ``lowerLeft``/``upperRight`` pixel corners to 0-based boxes."""
    boxes = []
    for roi in pixel_rois:
        try:
            boxes.append(
                {
                    "row0": int(roi["lowerLeftFramePix"][1]) - 1,
                    "col0": int(roi["lowerLeftFramePix"][0]) - 1,
                    "row1": int(roi["upperRightFramePix"][1]),
                    "col1": int(roi["upperRightFramePix"][0]),
                }
            )
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return boxes


def _breakview_boxes(unit) -> list[dict]:
    """ROI boxes from ``BreakViewJSON`` (ribbon scans)."""
    doc = _json_attr(unit, "BreakViewJSON")
    if not doc:
        return []
    return _boxes_from(doc.get("measurementROIMaps", []))


def _coordmap_boxes(unit) -> list[dict]:
    """ROI boxes from ``CoordinateMapJSON`` (linescan / multiline)."""
    doc = _json_attr(unit, "CoordinateMapJSON")
    if not doc:
        return []
    try:
        return _boxes_from(doc["maps"][0]["measurementROIs"])
    except (KeyError, IndexError, TypeError):
        return []


def _extend_to_rois(value, n_rois):
    """Broadcast a scalar / single list to one entry per ROI."""
    if not isinstance(value, list):
        return [value] * n_rois
    if value and not isinstance(value[0], list):
        return [value] * n_rois
    return value


def _quaternion_from_guideline(guideline) -> list[float]:
    """Rotation quaternion (w, x, y, z) taking +X onto the guideline direction."""
    start = np.array([guideline[0][0], guideline[1][0], guideline[2][0]], dtype=float)
    end = np.array([guideline[0][1], guideline[1][1], guideline[2][1]], dtype=float)
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return [1.0, 0.0, 0.0, 0.0]
    direction /= norm

    axis = np.cross(np.array([1.0, 0.0, 0.0]), direction)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0:
        return [1.0, 0.0, 0.0, 0.0]
    axis /= axis_norm

    angle = float(np.arccos(np.clip(float(np.dot([1.0, 0.0, 0.0], direction)), -1, 1)))
    s = math.sin(angle / 2)
    return [
        math.cos(angle / 2),
        float(axis[0] * s),
        float(axis[1] * s),
        float(axis[2] * s),
    ]


def _guideline_center(guideline, drift_length=None) -> list[float]:
    """Midpoint of a guideline, shortened to ``drift_length`` when given."""
    start = np.array([guideline[0][0], guideline[1][0], guideline[2][0]], dtype=float)
    end = np.array([guideline[0][1], guideline[1][1], guideline[2][1]], dtype=float)
    if drift_length is None:
        return ((start + end) / 2).tolist()
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return start.tolist()
    direction /= norm
    return ((start + start + direction * drift_length) / 2).tolist()


def _spatial_info(unit, modality: int) -> dict:
    """ROI centroids, rotations and pixel size for one unit.

    Ported from ``lab4.convert.aod.parse_imaging_info``. Returns
    ``{"centroids": [...], "rotations": [...], "pixel_size_um": float|None}``
    with empty lists when the unit declares no multi-ROI scan pattern.
    """
    blank = {"centroids": [], "rotations": [], "pixel_size_um": None}
    pattern = _scan_pattern(unit)
    if pattern is None:
        return blank

    def _as_points(points):
        pts = np.asarray(points).tolist()
        return pts if pts and isinstance(pts[0], list) else [pts]

    try:
        if modality in (6, 9):  # guideline-defined: linescan, ribbon transverse
            centroids = _as_points(np.mean(pattern["guideLine"], axis=2).tolist())
            key = "pixelSize" if modality == 6 else "pixelSizeL"
            return {
                "centroids": centroids,
                "rotations": [
                    _quaternion_from_guideline(g) for g in pattern["guideLine"]
                ],
                "pixel_size_um": _extend_to_rois(pattern[key], len(centroids))[0],
            }
        if modality == 10:  # ribbon longitudinal: guidelines shortened by drift
            drift = pattern.get("driftLength")
            centroids = [_guideline_center(g, drift) for g in pattern["guideLine"]]
            return {
                "centroids": centroids,
                "rotations": [
                    _quaternion_from_guideline(g) for g in pattern["guideLine"]
                ],
                "pixel_size_um": _extend_to_rois(
                    pattern["pixelSizeL"], len(centroids)
                )[0],
            }
        if modality == 7:  # multiline
            centroids = _as_points(np.asarray(pattern["centerPoints"]).T.tolist())
            return {
                "centroids": centroids,
                "rotations": [],
                "pixel_size_um": _extend_to_rois(
                    pattern["pixelSize"], len(centroids)
                )[0],
            }
        if modality == 8:  # chessboard
            centroids = _as_points(np.asarray(pattern["centerPoints"]).T.tolist())
            n = len(centroids)
            return {
                "centroids": centroids,
                "rotations": _extend_to_rois(pattern["rotation"]["e"], n),
                "pixel_size_um": _extend_to_rois(pattern["pixelSizeX"], n)[0],
            }
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.debug(
            f"scan pattern for {unit.name} (MethodType {modality}) unusable: {e}"
        )
        return blank
    return blank


def _parse_curves(unit) -> dict[str, dict]:
    """Timing curves for one unit, keyed by curve name.

    Each entry is ``{"timestamps": ms array, "values": raw array}``. Curves are
    small (one sample per scanner event), so they are read eagerly.
    """
    curves: dict[str, dict] = {}
    for key in unit:
        if not key.startswith("Curve_"):
            continue
        curve = unit[key]
        try:
            name = _attr(curve, "Name")
            if name not in CURVE_NAMES:
                continue
            delta = float(curve.attrs["CurveDataXRawDelta"])
            ts = np.roll(curve["CurveDataYIdxNextSample"][:], 1)
            ts[0] = 0
            curves[name] = {
                "timestamps": ts * delta,  # ms
                "values": curve["CurveDataYRawData"][:],
            }
        except (KeyError, TypeError, ValueError):
            continue
    return curves


def _find_sync_frame(curves, frame_period_ms, sync_key, sync_edge) -> int:
    """Frame index of the sync event, or 0 when it can't be located."""
    curve = curves.get(sync_key)
    if curve is None:
        return 0
    ts, vs = curve["timestamps"], curve["values"]
    if len(ts) < 2 or not frame_period_ms:
        return 0
    if sync_edge == "falling":
        if vs[0] == 1:
            time_ms = ts[1]  # first falling edge
        elif len(ts) > 2:
            time_ms = ts[2]  # curve opened low; the second event is the fall
        else:
            return 0
    elif vs[0] == 0:
        time_ms = ts[1]
    else:
        return 0
    return int(math.ceil(float(time_ms) / float(frame_period_ms)))


class _Layout:
    """Resolved read plan for one MUnit.

    Built once at open time from the unit's ``MethodType`` *and* the actual
    dataset shape -- the two disagree in the wild (``MethodType`` 7 is
    documented as framed but is written packed), and the shape is what a read
    has to satisfy.
    """

    __slots__ = (
        "kind",
        "nt",
        "nc",
        "nz",
        "ny",
        "nx",
        "rois",
        "flip_y",
        "z_meaning",
        "frame_maps",
        "raw_shape",
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))

    @property
    def light_paths(self) -> int:
        """Interleaved light paths per scanner frame; 1 unless dichroic switching.

        With switching on, one timepoint of this layout consumes `light_paths`
        raw scanner frames -- the divisor between anything expressed in
        ``TStepInMs`` units and anything expressed in this layout's T axis.
        """
        return len(self.frame_maps) if self.frame_maps else 1

    def describe(self) -> str:
        return (
            f"kind={self.kind} raw={self.raw_shape} -> "
            f"(T={self.nt}, C={self.nc}, Z={self.nz}, Y={self.ny}, X={self.nx}) "
            f"z={self.z_meaning} flip_y={self.flip_y}"
        )


def _resolve_layout(unit, modality: int, curves: dict, flip_y=None) -> _Layout:
    """Work out how to turn this unit's raw channels into canonical TCZYX."""
    nc = int(_attr(unit, "VecChannelsSize", 0) or 0)
    if nc <= 0:
        nc = sum(1 for k in unit if k.startswith("Channel_"))
    nc = max(1, nc)

    raw = unit["Channel_0"].shape
    if len(raw) != 3:
        raise ValueError(
            f"{unit.name}/Channel_0 has shape {raw}; MESc channels are expected "
            f"to be 3D (axis0, Y, X)."
        )
    n0, page_y, page_x = (int(v) for v in raw)

    if flip_y is None:
        flip_y = _FLIP_Y_BY_MODALITY.get(modality, False)
    common = {"nc": nc, "flip_y": bool(flip_y), "raw_shape": raw, "frame_maps": None}

    if modality == 2:
        # axis 0 is depth; there is no time axis at all.
        return _Layout(
            kind="zstack",
            nt=1,
            nz=n0,
            ny=page_y,
            nx=page_x,
            rois=[],
            z_meaning="depth",
            **common,
        )

    boxes = _breakview_boxes(unit) or _coordmap_boxes(unit)

    # Frames packed into Y: one raw "frame" holding T*n_lines rows. Detected
    # from the shape rather than the modality table, which disagrees.
    if modality in _PACKED_MODALITIES and boxes and n0 == 1:
        rois = []
        for box in boxes:
            n_lines = box["row1"] - box["row0"]
            width = box["col1"] - box["col0"]
            if n_lines <= 0 or width <= 0:
                continue
            rois.append(
                {
                    **box,
                    "n_lines": n_lines,
                    "width": width,
                    "nframes": page_y // n_lines,
                }
            )
        if rois:
            nt = min(r["nframes"] for r in rois)
            frame_maps = None
            # Dichroic switching interleaves light paths: green frames land on
            # Channel_0, red on Channel_1, at different timepoints. Each channel
            # therefore reads a different subset of source frames, and the
            # effective T is roughly halved.
            dichro = curves.get("DichroSw_AO1")
            if modality == 7 and dichro is not None and nc >= 2:
                values = np.asarray(dichro["values"])
                green = np.flatnonzero(values == _GREEN_LP)
                red = np.flatnonzero(values == _RED_LP)
                green, red = green[green < nt], red[red < nt]
                n_pairs = min(len(green), len(red))
                if n_pairs:
                    frame_maps = {0: green[:n_pairs], 1: red[:n_pairs]}
                    nt = n_pairs
                    nc = 2
            return _Layout(
                kind="packed",
                nt=nt,
                nz=len(rois),
                ny=max(r["n_lines"] for r in rois),
                nx=max(r["width"] for r in rois),
                rois=rois,
                z_meaning="roi_index",
                **{**common, "nc": nc, "frame_maps": frame_maps},
            )

    if modality in _BOX_MODALITIES and boxes:
        rois = [b for b in boxes if b["row1"] > b["row0"] and b["col1"] > b["col0"]]
        if rois:
            return _Layout(
                kind="boxes",
                nt=n0,
                nz=len(rois),
                ny=max(r["row1"] - r["row0"] for r in rois),
                nx=max(r["col1"] - r["col0"] for r in rois),
                rois=rois,
                z_meaning="roi_index",
                **common,
            )

    if modality in _TILED_MODALITIES or (modality in _BOX_MODALITIES and not boxes):
        # ROIs occupy contiguous, equal-width column blocks of the raw page.
        # A single-ROI scan has nothing to unpack — the page already is the
        # ROI — so it falls through to the framed read below without comment.
        n_rois = len(_spatial_info(unit, modality)["centroids"]) or 1
        if n_rois > 1:
            if page_x % n_rois == 0:
                x_per = page_x // n_rois
                rois = [
                    {
                        "row0": 0,
                        "row1": page_y,
                        "col0": i * x_per,
                        "col1": (i + 1) * x_per,
                    }
                    for i in range(n_rois)
                ]
                return _Layout(
                    kind="tiled",
                    nt=n0,
                    nz=n_rois,
                    ny=page_y,
                    nx=x_per,
                    rois=rois,
                    z_meaning="roi_index",
                    **common,
                )
            logger.warning(
                f"{unit.name}: MethodType {modality} declares {n_rois} ROIs but "
                f"the {page_x}px page doesn't divide evenly; reading as a "
                f"single FOV."
            )

    # timeseries, multicube, and every unrecognised modality: axis 0 is time,
    # the page is one FOV.
    if modality not in MODALITY_NAMES:
        logger.warning(
            f"{unit.name}: unknown MethodType {modality}; reading axis 0 as time."
        )
    return _Layout(
        kind="frames",
        nt=n0,
        nz=1,
        ny=page_y,
        nx=page_x,
        rois=[],
        z_meaning="none",
        **common,
    )


def _iso_time(posix_seconds) -> str | None:
    """ISO-8601 acquisition time, or None when the attribute is missing."""
    if posix_seconds in (None, 0):
        return None
    try:
        return datetime.fromtimestamp(float(posix_seconds), _ACQ_TZ).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def list_mesc_units(path: Path | str) -> list[dict]:
    """Describe every measurement unit in a ``.mesc`` file.

    Opens the file read-only and reads headers only -- no pixel data.

    Parameters
    ----------
    path : Path or str
        Path to the ``.mesc`` file.

    Returns
    -------
    list of dict
        One entry per ``MSession_N/MUnit_M`` holding a ``Channel_0``, in file
        order, with keys ``session``, ``munit``, ``key``, ``index``,
        ``modality``, ``modality_name``, ``kind``, ``shape`` (canonical TCZYX),
        ``nframes``, ``nchannels``, ``nrois``, ``comment`` and ``start_time``.

    Examples
    --------
    >>> for u in list_mesc_units("scan.mesc"):  # doctest: +SKIP
    ...     print(u["key"], u["modality_name"], u["shape"])
    """
    path = Path(path)
    units: list[dict] = []
    with h5py.File(path, "r") as f:
        sessions = sorted(
            (k for k in f if k.startswith("MSession")), key=_unit_sort_key
        )
        for session_key in sessions:
            session = f[session_key]
            for munit_key in sorted(
                (k for k in session if k.startswith("MUnit")), key=_unit_sort_key
            ):
                unit = session[munit_key]
                if "Channel_0" not in unit:
                    continue
                modality = int(_attr(unit, "MethodType", 0) or 0)
                try:
                    layout = _resolve_layout(unit, modality, _parse_curves(unit))
                except (ValueError, KeyError) as e:
                    logger.warning(f"skipping {session_key}/{munit_key}: {e}")
                    continue
                units.append(
                    {
                        "session": session_key,
                        "munit": munit_key,
                        "key": f"{session_key}/{munit_key}",
                        "index": len(units),
                        "modality": modality,
                        "modality_name": MODALITY_NAMES.get(
                            modality, f"unknown_{modality}"
                        ),
                        "kind": layout.kind,
                        "shape": (
                            layout.nt,
                            layout.nc,
                            layout.nz,
                            layout.ny,
                            layout.nx,
                        ),
                        "nframes": layout.nt,
                        "nchannels": layout.nc,
                        "nrois": layout.nz if layout.z_meaning == "roi_index" else 1,
                        "comment": _attr(unit, "Comment", "") or "",
                        "start_time": _iso_time(_attr(unit, "MeasurementDatePosix")),
                    }
                )
    return units


def _take_axis0(dataset, indices) -> np.ndarray:
    """Read ``dataset[indices]`` along axis 0 in as few h5py selections as possible.

    h5py fancy selection requires strictly increasing indices, so arbitrary
    (repeated, out-of-order) index lists are read sorted-unique and reordered
    in memory.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    if idx.size == 1:
        return dataset[int(idx[0])][None, ...]
    if np.all(np.diff(idx) == 1):
        return dataset[int(idx[0]) : int(idx[-1]) + 1]
    uniq, inverse = np.unique(idx, return_inverse=True)
    block = dataset[uniq.tolist()]
    if uniq.size == idx.size:
        return block
    return block[inverse]


class MescArray(RoiFeatureMixin, ReductionMixin, PhaseCorrectionMixin, Shape5DMixin):
    """
    Lazy reader for one measurement unit of a Femtonics ``.mesc`` file.

    Presents the canonical mbo 5D shape ``(T, C, Z, Y, X)``. For AOD multi-ROI
    scans the ``Z`` axis carries the ROI index (see the module docstring); for
    ``MethodType`` 2 it carries real depth; otherwise it is a singleton.

    Multi-ROI units also expose the standard ROI interface, so ``roi=None``
    puts every ROI together on one array, ``roi=N`` selects one, and ``roi=0``
    splits them -- the same semantics ScanImage mROI data has, which is what
    makes per-ROI viewer subplots and per-ROI export work unchanged.

    Parameters
    ----------
    filenames : Path or str
        Path to the ``.mesc`` file.
    session : str, optional
        Session group name (e.g. ``"MSession_0"``). Only used to qualify a bare
        ``munit``/``unit`` name.
    munit : str, optional
        Unit group name (e.g. ``"MUnit_3"``).
    unit : int or str, optional
        Unit selector: an index into `list_mesc_units`, or a full
        ``"MSession_0/MUnit_3"`` key. Overrides ``session``/``munit``.
        Defaults to the first unit in the file.
    roi : int or sequence of int, optional
        ROI selection. ``None`` keeps every ROI on the Z axis (default), ``N``
        selects one ROI (1-based), ``0`` splits all ROIs.
    flip_y : bool, optional
        Flip Y on read. Defaults to the ``lab4.convert`` behaviour: on for
        chessboard and both ribbon scans, off for everything else. Whether the
        flip also belongs on linescan/multiline is unsettled upstream; pass it
        explicitly to override.
    start_frame : int, optional
        Drop this many leading timepoints (crop-to-start alignment).
    sync_key : str, optional
        Timing curve whose edge marks t=0 (e.g. ``"DiI2"``). When given, the
        detected frame is used as ``start_frame``. The detected value is always
        reported as ``metadata["mesc_sync_frame"]``, applied or not -- in
        timepoints of this array, with the raw scanner-frame count it was
        derived from kept as ``metadata["mesc_sync_frame_raw"]``.
    sync_edge : {"falling", "rising"}, optional
        Edge to look for. Default ``"falling"``.
    fix_phase : bool, default False
        Apply bidirectional scan-phase correction on read. Off by default; the
        GUI's scan-phase controls toggle it.
    phasecorr_method, border, max_offset, use_fft
        Phase-correction settings, as on `ScanImageArray`.

    Attributes
    ----------
    units : list of dict
        Every unit in the file (see `list_mesc_units`).
    unit_key : str
        The ``"MSession_N/MUnit_M"`` this array reads.
    modality : int
        The unit's ``MethodType``.

    Examples
    --------
    >>> arr = MescArray("scan.mesc", unit=2)           # doctest: +SKIP
    >>> arr.shape                                       # doctest: +SKIP
    (1200, 2, 6, 64, 96)
    >>> arr.metadata["mesc_z_axis_meaning"]             # doctest: +SKIP
    'roi_index'
    """

    PRIORITY = 60

    METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Height of one ROI (padded to the largest ROI for ribbon scans).",
        "Lx": "Width of one ROI after unpacking from the raw MESc page.",
        "num_zplanes": (
            "ROI count for AOD multi-ROI scans, depth for MethodType 2 z-stacks. "
            "See mesc_z_axis_meaning."
        ),
        "dz": (
            "Z-step. None for AOD multi-ROI -- those ROIs are not a uniform "
            "depth series."
        ),
        "fs": (
            "Rate of the timepoints this array reports. Halved (per light path) "
            "for dichroic multiline; see mesc_raw_frame_rate for the scanner's."
        ),
    }

    def __init__(
        self,
        filenames: Path | str,
        session: str | None = None,
        munit: str | None = None,
        unit: int | str | None = None,
        roi: int | Sequence[int] | None = None,
        flip_y: bool | None = None,
        start_frame: int | None = None,
        sync_key: str | None = None,
        sync_edge: str | None = None,
        fix_phase: bool = False,
        phasecorr_method: str = "mean",
        border: int = 3,
        max_offset: int = 4,
        use_fft: bool = True,
    ):
        self.filenames = [Path(filenames)]
        path = self.filenames[0]
        self._f = h5py.File(path, "r")
        self._metadata_overlay: dict = {}

        self.units = list_mesc_units(path)
        if not self.units:
            self._f.close()
            raise ValueError(f"No readable MSession/MUnit groups in {path}")

        selected = self._select_unit(unit, session, munit)
        self.unit_key = selected["key"]
        self.modality = selected["modality"]
        self._unit = self._f[self.unit_key]
        self._curves = _parse_curves(self._unit)
        self._layout = _resolve_layout(
            self._unit, self.modality, self._curves, flip_y=flip_y
        )

        frame_period_ms = _attr(self._unit, "TStepInMs")
        self._frame_period_ms = float(frame_period_ms) if frame_period_ms else None
        # `TStepInMs` ticks once per *raw* scanner frame, so the sync search
        # lands on a raw frame index. Dichroic switching makes one layout
        # timepoint span several raw frames, so convert before it is used as a
        # crop -- rounding up keeps the crop at or after the sync edge.
        self._sync_frame_raw = _find_sync_frame(
            self._curves,
            self._frame_period_ms,
            sync_key or SYNC_KEY_DEFAULT,
            sync_edge or SYNC_EDGE_DEFAULT,
        )
        paths = self._layout.light_paths
        self._sync_frame = math.ceil(self._sync_frame_raw / paths)
        self._start_frame = self._resolve_start_frame(start_frame, sync_key)
        self._nt = max(0, self._layout.nt - self._start_frame)

        self._dtype = self._unit["Channel_0"].dtype
        self._target_dtype = None
        self._channels = [
            self._unit[f"Channel_{c}"]
            for c in range(self._layout.nc)
            if f"Channel_{c}" in self._unit
        ]
        if len(self._channels) < self._layout.nc:
            logger.warning(
                f"{self.unit_key}: VecChannelsSize={self._layout.nc} but only "
                f"{len(self._channels)} Channel_N dataset(s) exist; using the latter."
            )
            self._layout.nc = max(1, len(self._channels))

        # ROI interface: one entry per Z slot when Z means "ROI".
        self._rois = self._build_roi_slices()
        self._roi = None
        if roi is not None:
            self.roi = roi

        self._offset_cache: dict[tuple[int, int, int], float] = {}
        self.phase_correction = PhaseCorrectionFeature(
            enabled=fix_phase,
            method=phasecorr_method,
            shift=None,
            use_fft=use_fft,
            border=border,
            max_offset=max_offset,
        )
        self.phase_correction.add_event_handler(
            lambda _event: self._invalidate_offset_cache()
        )

        self._metadata = self._build_metadata()
        logger.info(
            f"{path.name} [{self.unit_key}] {self._layout.describe()} "
            f"start_frame={self._start_frame}"
        )

    # -- construction helpers -------------------------------------------

    def _select_unit(self, unit, session, munit) -> dict:
        """Resolve the unit selector against the file's unit list."""
        if unit is None and munit is not None:
            unit = f"{session or 'MSession_0'}/{munit}"

        if unit is None:
            if len(self.units) > 1:
                logger.warning(
                    f"{self.filenames[0].name} holds {len(self.units)} units; "
                    f"opening {self.units[0]['key']}. Pass unit=<index or key> "
                    f"(CLI: --unit) to choose another; `mbo info` lists them."
                )
            return self.units[0]

        if isinstance(unit, (int, np.integer)):
            if not 0 <= int(unit) < len(self.units):
                raise ValueError(
                    f"unit index {unit} out of range; {self.filenames[0].name} has "
                    f"{len(self.units)} unit(s): {[u['key'] for u in self.units]}"
                )
            return self.units[int(unit)]

        key = str(unit)
        if "/" not in key:
            key = f"{session or 'MSession_0'}/{key}"
        for entry in self.units:
            if entry["key"] == key:
                return entry
        raise ValueError(
            f"unit {key!r} not found in {self.filenames[0].name}; "
            f"available: {[u['key'] for u in self.units]}"
        )

    def _resolve_start_frame(self, start_frame, sync_key) -> int:
        """Leading timepoints to drop, from an explicit count or a sync curve."""
        if start_frame is None and sync_key is None:
            return 0
        value = int(start_frame) if start_frame is not None else self._sync_frame
        if value < 0:
            raise ValueError(f"start_frame must be >= 0, got {value}")
        if value >= self._layout.nt:
            logger.warning(
                f"{self.unit_key}: start_frame {value} >= {self._layout.nt} "
                f"timepoints; not cropping."
            )
            return 0
        return value

    def _build_roi_slices(self) -> list[dict]:
        """ROI geometry in the `RoiFeatureMixin` shape; empty when Z isn't ROIs."""
        if self._layout.z_meaning != "roi_index":
            return []
        out = []
        for i, roi in enumerate(self._layout.rois):
            height = roi.get("n_lines", roi["row1"] - roi["row0"])
            width = roi.get("width", roi["col1"] - roi["col0"])
            # After padding to the array's (ny, nx) and the Y flip, valid rows
            # sit at the bottom of the frame; record where they actually land.
            y0 = self._layout.ny - height if self._layout.flip_y else 0
            out.append(
                {
                    "index": i,
                    "y_start": y0,
                    "y_end": y0 + height,
                    "x": 0,
                    "height": height,
                    "width": width,
                    "slice": slice(y0, y0 + height),
                    "source_row0": roi["row0"],
                    "source_col0": roi["col0"],
                }
            )
        return out

    def _build_metadata(self) -> dict:
        """Canonical metadata for this unit, plus the MESc-specific extras."""
        layout = self._layout
        info = _spatial_info(self._unit, self.modality)
        pixel_size = info["pixel_size_um"]
        # `TStepInMs` is the raw scanner frame period. Dichroic switching hands
        # each light path every Nth raw frame, so a channel's own timepoints
        # arrive N times slower than the scanner runs -- and it is those
        # timepoints the T axis counts. Reporting the raw rate here would make
        # `num_timepoints / fs` claim half the duration actually recorded.
        raw_fs = 1000.0 / self._frame_period_ms if self._frame_period_ms else None
        fs = raw_fs / layout.light_paths if raw_fs else None

        if layout.frame_maps is not None:
            channel_names = ["Green", "Red"]
        else:
            defaults = ["Green", "Red"]
            channel_names = [
                str(_attr(self._unit.get(f"Channel_{c}", self._unit), "Name") or "")
                or (defaults[c] if c < len(defaults) else f"Channel_{c}")
                for c in range(layout.nc)
            ]

        md: dict = {
            # canonical
            "fs": fs,
            "frame_rate": fs,
            "dx": float(pixel_size) if pixel_size else None,
            "dy": float(pixel_size) if pixel_size else None,
            # AOD ROIs are not a uniform depth series, and MESc records no z-step
            # we can trust for a real stack either, so dz stays user-supplied.
            "dz": None,
            "num_timepoints": self._nt,
            "num_zplanes": layout.nz,
            "nchannels": layout.nc,
            "num_color_channels": layout.nc,
            "Ly": layout.ny,
            "Lx": layout.nx,
            "dtype": np.dtype(self._dtype).name,
            "num_mrois": layout.nz if layout.z_meaning == "roi_index" else 1,
            # mesc specifics
            "mesc_file": str(self.filenames[0]),
            "mesc_unit": self.unit_key,
            "mesc_unit_index": next(
                (u["index"] for u in self.units if u["key"] == self.unit_key), 0
            ),
            "mesc_units_in_file": len(self.units),
            "mesc_modality": self.modality,
            "mesc_modality_name": MODALITY_NAMES.get(
                self.modality, f"unknown_{self.modality}"
            ),
            "mesc_layout": layout.kind,
            "mesc_raw_shape": tuple(layout.raw_shape),
            "mesc_z_axis_meaning": layout.z_meaning,
            "mesc_flip_y": layout.flip_y,
            "mesc_centroids": info["centroids"],
            "mesc_rotations": info["rotations"],
            "mesc_roi_extents": [
                {k: r[k] for k in ("index", "y_start", "y_end", "height", "width")}
                for r in self._rois
            ],
            "mesc_sync_frame": self._sync_frame,
            "mesc_sync_frame_raw": self._sync_frame_raw,
            "mesc_start_frame": self._start_frame,
            "mesc_raw_frame_rate": raw_fs,
            "mesc_light_paths": layout.light_paths,
            "mesc_curves": sorted(self._curves),
            "mesc_dichroic": layout.frame_maps is not None,
            "channel_names": channel_names,
            "comment": _attr(self._unit, "Comment", "") or "",
            "start_time": _iso_time(_attr(self._unit, "MeasurementDatePosix")),
            "start_time_tz": "UTC-05:00",
        }
        # raw unit attrs, minus the large embedded JSON documents
        md["mesc_attrs"] = {
            k: _attr(self._unit, k) for k in self._unit.attrs if k not in _JSON_ATTRS
        }
        return md

    # -- identity / dispatch --------------------------------------------

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """True for ``.mesc`` files that are actually HDF5 containers."""
        if not file or not isinstance(file, (str, Path)):
            return False
        path = Path(file)
        if path.suffix.lower() != ".mesc" or not path.is_file():
            return False
        try:
            return bool(h5py.is_hdf5(path))
        except OSError:
            return False

    @property
    def reader_kwargs(self) -> dict:
        """Kwargs `imread` needs to re-open this exact unit in another process."""
        return {"unit": self.unit_key}

    # -- shape / dtype ---------------------------------------------------

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        layout = self._layout
        nz = layout.nz
        if isinstance(self.roi, (int, np.integer)) and self.roi > 0:
            nz = 1
        return (self._nt, layout.nc, nz, layout.ny, layout.nx)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return self._shape5d()

    @property
    def num_planes(self) -> int:
        """Size of the Z axis (ROI count or depth -- see ``mesc_z_axis_meaning``)."""
        return self._shape5d()[2]

    @property
    def num_color_channels(self) -> int:
        """Number of detector channels."""
        return self._layout.nc

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set the target dtype for lazy conversion on read."""
        self._target_dtype = np.dtype(dtype)
        return self

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def slider_dim_labels(self) -> tuple[str, ...]:
        """Viewer slider labels for the non-singleton T/C/Z axes.

        Reports the shape of what the viewer will actually render: with a
        split-ROI selection (``roi=0`` or a list) each subplot holds one ROI,
        so Z is a singleton there even though this array still spans all of
        them. Getting that wrong desynchronises fastplotlib's slider count
        from the arrays it is handed.
        """
        nt, nc, nz, _, _ = self._shape5d()
        if self.roi is not None:
            nz = 1  # every rendered subplot carries a single ROI
        z_label = {"roi_index": "ROI", "depth": "Z-plane"}.get(
            self._layout.z_meaning, "Z"
        )
        labels = []
        if nt > 1:
            labels.append("Timepoint")
        if nc > 1:
            labels.append("Channel")
        if nz > 1:
            labels.append(z_label)
        return tuple(labels)

    # -- metadata --------------------------------------------------------

    @property
    def metadata(self) -> dict:
        """Unit metadata merged with any in-memory overrides. Always a dict."""
        md = dict(self._metadata)
        md.update(self._metadata_overlay)
        # dimension counts follow the current ROI selection, not the file's
        # full extent — a single-ROI view really is one plane deep, and the
        # writer reads these keys rather than re-deriving them from shape.
        nt, nc, nz, ny, nx = self._shape5d()
        md.update(
            {
                "num_timepoints": nt,
                "nchannels": nc,
                "num_color_channels": nc,
                "num_zplanes": nz,
                "Ly": ny,
                "Lx": nx,
                "fix_phase": self.fix_phase,
                "phasecorr_method": self.phasecorr_method,
                "offset": self.offset,
                "border": self.border,
                "max_offset": self.max_offset,
                "use_fft": self.use_fft,
                "roi": self.roi,
                "roi_mode": self.roi_mode,
            }
        )
        return md

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        # the file is opened read-only; overrides live in memory.
        self._metadata_overlay.update(value)

    # -- reading ---------------------------------------------------------

    def _source_frames(self, c: int, frames: Sequence[int]) -> np.ndarray:
        """Map requested timepoints onto source frame indices for channel `c`."""
        idx = np.asarray(frames, dtype=np.int64) + self._start_frame
        maps = self._layout.frame_maps
        if maps is not None and c in maps:
            return np.asarray(maps[c], dtype=np.int64)[idx]
        return idx

    def _read_block(self, c: int, z: int, frames: Sequence[int]) -> np.ndarray:
        """Read one (channel, z) column as ``(len(frames), ny, nx)``."""
        layout = self._layout
        dataset = self._channels[min(c, len(self._channels) - 1)]
        kind = layout.kind

        if kind == "zstack":
            # axis 0 is depth; T is a singleton, so every timepoint is one plane
            plane = dataset[int(z)][None, ...]
            block = plane if len(frames) == 1 else np.repeat(plane, len(frames), axis=0)
        elif kind == "packed":
            block = self._read_packed(dataset, c, z, frames)
        else:
            raw = _take_axis0(dataset, self._source_frames(c, frames))
            if kind == "frames":
                block = raw
            else:  # tiled / boxes: crop the ROI box off the page
                roi = layout.rois[z]
                block = raw[:, roi["row0"] : roi["row1"], roi["col0"] : roi["col1"]]

        block = np.ascontiguousarray(block)
        if block.shape[1:] != (layout.ny, layout.nx):
            block = self._pad_to_frame(block)
        if layout.flip_y:
            block = block[:, ::-1, :]
        if self.fix_phase and block.size:
            block = self._apply_phase(block, c, z, frames)
        return block

    def _read_packed(self, dataset, c, z, frames) -> np.ndarray:
        """Unpack frames stored as consecutive row blocks of a single raw page."""
        roi = self._layout.rois[z]
        n_lines, col0, col1 = roi["n_lines"], roi["col0"], roi["col1"]
        src = self._source_frames(c, frames)
        if src.size == 0:
            return np.empty((0, n_lines, col1 - col0), dtype=dataset.dtype)
        if src.size > 1 and np.all(np.diff(src) == 1):
            rows = dataset[
                0, int(src[0]) * n_lines : (int(src[-1]) + 1) * n_lines, col0:col1
            ]
            return rows.reshape(src.size, n_lines, col1 - col0)
        out = np.empty((src.size, n_lines, col1 - col0), dtype=dataset.dtype)
        for i, f in enumerate(src):
            out[i] = dataset[0, int(f) * n_lines : (int(f) + 1) * n_lines, col0:col1]
        return out

    def _pad_to_frame(self, block: np.ndarray) -> np.ndarray:
        """Zero-pad a ROI smaller than the array's frame to the common size.

        ROIs of a ribbon or linescan unit differ in size; the array reports the
        largest. Padded pixels are indistinguishable from real dark ones once
        written, so ``metadata["mesc_roi_extents"]`` records where each ROI's
        valid region lands.
        """
        layout = self._layout
        out = np.zeros((block.shape[0], layout.ny, layout.nx), dtype=block.dtype)
        h = min(block.shape[1], layout.ny)
        w = min(block.shape[2], layout.nx)
        out[:, :h, :w] = block[:, :h, :w]
        return out

    def _apply_phase(self, block, c, z, frames) -> np.ndarray:
        """Bidirectional scan-phase correction, mirroring `ScanImageArray`."""
        shift = self.phase_correction.effective_shift
        if shift is not None:
            corrected, offset = _apply_offset(block, shift, use_fft=self.use_fft), shift
        else:
            corrected, offset = bidir_phasecorr(
                block,
                method=self.phasecorr_method,
                max_offset=self.max_offset,
                border=self.border,
                use_fft=self.use_fft,
            )
        for t in frames:
            self._offset_cache[(int(t), int(c), int(z))] = float(offset)
        return corrected

    def get_offset_at(self, t: int, c: int = 0, z: int = 0) -> float | None:
        """Cached scan-phase offset for one (t, c, z) cell, or None."""
        return self._offset_cache.get((int(t), int(c), int(z)))

    def _invalidate_offset_cache(self) -> None:
        self._offset_cache.clear()

    def _z_indices(self, z_key) -> list[int]:
        """Requested Z positions mapped onto layout ROI/plane indices."""
        selected = listify_index(z_key, self._shape5d()[2])
        roi = self.roi
        if isinstance(roi, (int, np.integer)) and roi > 0:
            # a single-ROI view exposes one Z slot, backed by ROI `roi`
            return [int(roi) - 1 for _ in selected]
        return [int(z) for z in selected]

    def __getitem__(self, key):
        nt, nc, _nz, ny, nx = self._shape5d()
        key = _normalize_key(key, 5)
        key = tuple(
            slice(k.start, k.stop, k.step) if isinstance(k, range) else k for k in key
        )
        key = key + (slice(None),) * (5 - len(key))
        t_key, c_key, z_key, y_key, x_key = key

        frames = listify_index(t_key, nt)
        colors = listify_index(c_key, nc)
        zs = self._z_indices(z_key)
        if not frames or not colors or not zs:
            return np.empty((0,), dtype=self.dtype)

        out = np.empty((len(frames), len(colors), len(zs), ny, nx), dtype=self._dtype)
        for ci, c in enumerate(colors):
            for zi, z in enumerate(zs):
                out[:, ci, zi] = self._read_block(int(c), int(z), frames)

        out = out[:, :, :, y_key]
        out = out[..., x_key]

        squeeze = tuple(i for i in range(3) if isinstance(key[i], (int, np.integer)))
        if squeeze:
            out = np.squeeze(out, axis=squeeze)
        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def __array__(self, dtype=None, copy=None):
        # one representative (Y, X) frame -- never an accidental full load
        data = np.asarray(self[0, 0, 0])
        if self._target_dtype is not None:
            data = data.astype(self._target_dtype)
        return data.astype(dtype) if dtype is not None else data

    def close(self):
        """Close the underlying HDF5 file."""
        self._f.close()

    def __repr__(self) -> str:
        return (
            f"MescArray({self.filenames[0].name!r}, unit={self.unit_key!r}, "
            f"modality={self._metadata.get('mesc_modality_name')!r}, "
            f"shape={self.shape}, dtype={np.dtype(self.dtype).name})"
        )

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        """Write this unit to disk in any supported output format."""
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            **kwargs,
        )


register_array_class(MescArray, priority=60)
