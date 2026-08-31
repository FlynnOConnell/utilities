"""Registration -> ROI subset -> extraction | demixing | discovery, on any lazy array.

The one thing everything here needs from a movie is

    arr[t, c, z, y, x]  ->  numpy

for integer / slice / list keys on every axis (rank 5, 4 ``TZYX``, 3 ``TYX``
or 2 ``YX`` arrays are all accepted). ``as_movie`` wraps such an array as a
``PlaneMovie`` - a ``(T, Y, X)`` view over one channel and one z-plane that
forwards ``movie[t, y0:y1, x0:x1]`` straight to the array - and every
consumer (mean extraction, suite2p's extractor, masknmf compression, quick
traces) reads through that view. If the array is spatially lazy, ROI reads
touch only the ROI's bounding box; if it is not, they still work.

Sources can be a lazy array, a numpy array, or anything ``imread`` opens -
a raw file, a suite2p / masknmf plane dir (``ops.npy`` + registered
``data.bin``), or a directory of plane dirs.

ROIs come from the manual-ROI tool (``manual_labels.zarr`` /
``RoiLabelStore``). The processing steps are

- **extract** - suite2p-style trace extraction with the drawn masks (no
  detection): ``F`` / ``Fneu`` / ``stat`` / ``iscell``;
- **demix** - masknmf NMF seeded with the drawn masks as initial spatial
  footprints, writing the same suite2p-shaped sidecars plus
  ``demixing_results.hdf5``;
- **discover** - unseeded detection inside a rectangular region: masknmf's
  superpixel initialisation or suite2p's detector runs on the crop and the
  outputs are written back in full-frame coordinates.

Outputs go to ``<out_dir>/`` - by default ``rois_<tag>/`` beside the source -
so a full-detection run in a plane dir is never overwritten; the subset's
``ops.npy`` points ``reg_file`` back at the parent ``data.bin`` when there is
one and ``roi_indices.npy`` maps each output row to its index in the store.

Draw the masks on the same frames the pipeline reads - the registered
``data.bin``. ``run(..., process="none")`` registers and stops;
``mbo <save_path>`` or ``mbo <plane_dir>`` then opens that registered movie
in the viewer, the ROI tool autosaves ``manual_labels.zarr`` into the
directory that was opened, and ``run(<save_path>, register_method="none")``
picks those stores up per plane (:func:`plane_store`).

Only numpy/scipy are required for the ``"mean"`` extraction engine and the
quick traces; suite2p, lbm_suite2p_python and masknmf are imported lazily
where used.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np

from mbo_utilities import log
from mbo_utilities.annotation.ngff import LabelsZarr
from mbo_utilities.annotation.store import RoiLabelStore

__all__ = [
    "PlaneMovie",
    "RoiSelection",
    "RunResult",
    "as_movie",
    "load_rois",
    "load_run_dir",
    "select_rois",
    "plane_masks",
    "roi_trace",
    "pixel_trace",
    "open_registered",
    "plane_index",
    "plane_store",
    "labels_path",
    "register",
    "extract_rois",
    "demix_rois",
    "discover_rois",
    "feather_mask",
    "pmd_crop",
    "run",
]

RegisterMethod = Literal["suite2p", "masknmf", "none"]
ProcessMethod = Literal["extract", "demix", "discover", "none"]
ExtractEngine = Literal["mean", "suite2p"]
DiscoverEngine = Literal["masknmf", "suite2p"]

OUT_PREFIX = "rois_"
SAVE_NAME = "manual_labels.zarr"  # same as gui.manual_roi.SAVE_NAME
_ZPLANE_RE = re.compile(r"zplane(\d+)")
_Z_RE = re.compile(r"z(\d+)")


def labels_path(fpath) -> Path:
    """Where ``fpath``'s annotations live: ``manual_labels.zarr`` beside a
    file, or inside a directory (mirrors ``gui.manual_roi.labels_path``
    without importing the GUI)."""
    base = Path.cwd() if fpath is None else Path(fpath)
    return (base.parent if base.suffix else base) / SAVE_NAME


# ---------------------------------------------------------------------------
# the movie view
# ---------------------------------------------------------------------------


def _index_len(key, n: int) -> int | None:
    """Length of axis ``n`` after indexing with ``key``; None when dropped."""
    if isinstance(key, (int, np.integer)):
        return None
    if isinstance(key, slice):
        return len(range(*key.indices(n)))
    if isinstance(key, range):
        return len(key)
    return len(np.asarray(key).reshape(-1))


def _shift(key, off: int, n: int):
    """``key`` on a length-``n`` cropped axis, moved by ``off`` into the source."""
    if isinstance(key, (int, np.integer)):
        k = int(key) + (n if key < 0 else 0)
        if not 0 <= k < n:
            raise IndexError(f"index {key} out of range for a length-{n} crop")
        return k + off
    if isinstance(key, slice):
        lo, hi, step = key.indices(n)
        if len(range(lo, hi, step)) == 0:
            return slice(0, 0)
        lo, hi = lo + off, hi + off
        return slice(lo, hi if hi >= 0 else None, step)
    a = np.asarray(key).reshape(-1)
    return np.where(a < 0, a + n, a) + off


_RANK_DIMS = {2: "YX", 3: "TYX", 4: "TZYX", 5: "TCZYX"}


def movie_dims(arr) -> tuple[str, ...]:
    """Axis names of ``arr``: its own ``dims`` when they match its rank and
    name Y and X, else the canonical letters for that rank."""
    nd = getattr(arr, "ndim", None)
    nd = int(np.ndim(arr) if nd is None else nd)
    dims = tuple(str(d).upper() for d in (getattr(arr, "dims", None) or ()))
    if len(dims) == nd and {"Y", "X"} <= set(dims):
        return dims
    if nd not in _RANK_DIMS:
        raise ValueError(f"movie source must be 2-5D, got ndim={nd}")
    return tuple(_RANK_DIMS[nd])


class PlaneMovie:
    """``(T, Y, X)`` view over one channel / z-plane of any indexable array.

    ``movie[t, y, x]`` is forwarded to the array by axis name (``arr.dims``,
    or T/C/Z/Y/X by rank), other axes pinned at ``z`` / ``c`` / 0, and the
    result is coerced to numpy and reshaped to what numpy indexing would
    give, so consumers never see a backend's squeeze quirks. This is all
    the ROI pipeline, masknmf compression and the viewer's traces rely on:
    if the array slices in y and x, everything here works. ``crop`` returns
    a rectangular view whose keys shift into the source frame on the way
    through; ``box`` records where that view sits.
    """

    y0 = 0
    x0 = 0

    def __init__(self, arr, z: int = 0, c: int = 0):
        self.arr = arr
        self.dims = movie_dims(arr)
        self.z = int(z)
        self.c = int(c)
        size = dict(zip(self.dims, (int(s) for s in arr.shape)))
        nt, nc, nz = size.get("T", 1), size.get("C", 1), size.get("Z", 1)
        if not 0 <= self.z < nz:
            raise IndexError(f"z={z} out of range for {nz} plane(s)")
        if not 0 <= self.c < nc:
            raise IndexError(f"c={c} out of range for {nc} channel(s)")
        self.shape = (nt, size["Y"], size["X"])
        self.nz, self.nc = nz, nc
        self.ndim = 3

    @property
    def dtype(self):
        return np.dtype(getattr(self.arr, "dtype", np.float32))

    def __len__(self):
        return self.shape[0]

    @property
    def box(self) -> tuple[int, int, int, int] | None:
        """``(y0, y1, x0, x1)`` of this view in the source frame; None when
        the view covers the whole frame."""
        _, ny, nx = self.shape
        size = dict(zip(self.dims, (int(s) for s in self.arr.shape)))
        if (self.y0, self.x0) == (0, 0) and (ny, nx) == (size["Y"], size["X"]):
            return None
        return (self.y0, self.y0 + ny, self.x0, self.x0 + nx)

    def crop(self, y0: int, y1: int, x0: int, x1: int) -> PlaneMovie:
        """View of rows ``y0:y1`` and columns ``x0:x1`` sharing this movie's
        array; crops of crops compose."""
        _, ny, nx = self.shape
        if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
            raise IndexError(f"crop ({y0}:{y1}, {x0}:{x1}) outside {ny}x{nx}")
        out = PlaneMovie(self.arr, z=self.z, c=self.c)
        out.y0, out.x0 = self.y0 + int(y0), self.x0 + int(x0)
        out.shape = (self.shape[0], int(y1 - y0), int(x1 - x0))
        return out

    def _full_key(self, t, y, x):
        if self.box is not None:
            _, ny, nx = self.shape
            y = _shift(y, self.y0, ny)
            x = _shift(x, self.x0, nx)
        axes = {"T": t, "C": self.c, "Z": self.z, "Y": y, "X": x}
        return tuple(axes.get(d, 0) for d in self.dims)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 3:
            raise IndexError(f"PlaneMovie takes at most 3 keys, got {len(key)}")
        t, y, x = (key + (slice(None),) * 3)[:3]
        if isinstance(t, range):
            t = slice(t.start, t.stop, t.step)
        if isinstance(t, (list, np.ndarray)):
            t = np.asarray(t).reshape(-1)
        nt, ny, nx = self.shape
        want = tuple(n for n in (_index_len(t, nt), _index_len(y, ny), _index_len(x, nx)) if n is not None)
        if "T" not in self.dims and not (isinstance(t, (int, np.integer)) or _index_len(t, 1) == 1):
            raise IndexError("a source without a time axis has a single frame")
        out = np.asarray(self.arr[self._full_key(t, y, x)])
        return out.reshape(want)

    def frames(self, t0: int, t1: int, y=slice(None), x=slice(None)) -> np.ndarray:
        """``(t1 - t0, ...)`` block, always with a leading time axis."""
        out = self[t0:t1, y, x]
        return out.reshape((-1,) + out.shape[-2:]) if out.ndim >= 2 else out.reshape(-1, 1, 1)


def as_movie(source, z: int = 0, c: int = 0) -> PlaneMovie:
    """``PlaneMovie`` for a lazy array, a numpy array or an ``imread``-able path."""
    if isinstance(source, PlaneMovie):
        return source
    if isinstance(source, (str, Path)):
        from mbo_utilities.reader import imread

        source = imread(source)
    return PlaneMovie(source, z=z, c=c)


def _source_nz(arr) -> int:
    return int(dict(zip(movie_dims(arr), arr.shape)).get("Z", 1))


# ---------------------------------------------------------------------------
# ROI selection
# ---------------------------------------------------------------------------


@dataclass
class RoiSelection:
    """Which ROIs of a store to run. Empty fields mean "no filter on this".

    ``planes`` are 0-based z indices, ``indices`` are store indices, and
    ``labels`` are class-label *names* from the store's label set. The
    criteria intersect, so ``planes=[0], labels=["soma"]`` is "somata on the
    first plane".
    """

    planes: list[int] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_rois(rois, *, source=None) -> RoiLabelStore:
    """Resolve ``rois`` to a ``RoiLabelStore``.

    ``rois`` may be a store, a path to a labels zarr, a path whose sibling
    ``manual_labels.zarr`` should be used, or ``None`` (then ``source`` is
    used the same way - the annotations the GUI saved next to the data).
    """
    if isinstance(rois, RoiLabelStore):
        return rois
    target = rois if rois is not None else source
    if target is None:
        raise ValueError("no ROI source: pass a RoiLabelStore, a labels zarr, or a data path")
    target = Path(target)
    if target.suffix != ".zarr" or not (target / "zarr.json").exists():
        target = labels_path(target)
    return LabelsZarr.load(target)


def select_rois(
    store: RoiLabelStore,
    planes: list[int] | None = None,
    indices: list[int] | None = None,
    labels: list[str] | None = None,
) -> list[int]:
    """Indices of the store's ROIs matching every given criterion.

    With no criteria this is every ROI. Unknown label names raise so a typo
    does not silently run zero ROIs.
    """
    keep = list(range(len(store.rois)))
    if indices:
        wanted = set(int(i) for i in indices)
        bad = sorted(i for i in wanted if not 0 <= i < len(store.rois))
        if bad:
            raise IndexError(f"ROI indices out of range for {len(store.rois)} ROIs: {bad}")
        keep = [i for i in keep if i in wanted]
    if planes:
        zs = set(int(z) for z in planes)
        keep = [i for i in keep if store.rois[i].z in zs]
    if labels:
        names = list(store.label_names)
        unknown = [n for n in labels if n not in names]
        if unknown:
            raise KeyError(f"unknown ROI labels {unknown}; store has {names}")
        classes = {names.index(n) for n in labels}
        keep = [i for i in keep if store.rois[i].class_index in classes]
    return keep


def plane_masks(
    store: RoiLabelStore, z: int, indices: list[int]
) -> tuple[np.ndarray, list[int]]:
    """``(label_image, kept)`` for the ROIs of ``indices`` that live on plane ``z``.

    ``label_image`` is ``(Y, X)`` uint16 with values ``1..K`` numbered in
    ``kept`` order (suite2p / cellpose mask convention); ``kept`` holds the
    store indices in that order.
    """
    kept = [i for i in indices if store.rois[i].z == int(z)]
    plane = store.labels[int(z)]
    out = np.zeros(plane.shape, np.uint16)
    for k, i in enumerate(kept, start=1):
        out[plane == (i + 1)] = k
    return out, kept


# ---------------------------------------------------------------------------
# quick traces
# ---------------------------------------------------------------------------


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("empty mask")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def feather_mask(mask: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """Soft-edged weights for a binary mask, 1 in the interior falling off
    over ``edge_width`` px toward the boundary (lbm_suite2p_python's
    feathering, kept inside the mask so trace weights never leave it)."""
    from scipy.ndimage import distance_transform_edt

    inside = distance_transform_edt(np.asarray(mask, bool))
    return np.clip(inside / max(int(edge_width), 1), 0.0, 1.0).astype(np.float32)


def _factorized(arr) -> bool:
    """True for masknmf factorized arrays, whose ``__getitem__`` reconstructs
    only the requested crop; frame batching then just adds overhead."""
    return callable(getattr(arr, "getitem_tensor", None))


def roi_trace(source, mask: np.ndarray, t=slice(None), *, z: int = 0, c: int = 0, batch: int = 500, weights: np.ndarray | None = None) -> np.ndarray:
    """Mean over ``mask`` per frame, reading only the mask's bounding box.

    Parameters
    ----------
    source
        Anything :func:`as_movie` accepts - a lazy array, a numpy array, a
        ``PlaneMovie`` or an ``imread``-able path.
    mask : np.ndarray
        ``(Y, X)`` boolean mask.
    t : slice or int, optional
        Frames to read; all by default.
    z, c : int, optional
        Plane and channel passed to :func:`as_movie`.
    batch : int, optional
        Frames per read on a raw movie, so a long recording never lands in
        RAM at once. A masknmf factorized array (``PMDArray``, ``ACArray``,
        ``ResidualArray``, ...) reconstructs only the bounding box, so it is
        read in a single call regardless of ``batch``.
    weights : np.ndarray, optional
        Full-frame weight image (e.g. :func:`feather_mask`); makes the trace
        a weighted mean over the mask's pixels.

    Returns
    -------
    np.ndarray
        ``(num_frames,)`` float32 trace.
    """
    movie = as_movie(source, z=z, c=c)
    if _factorized(movie.arr):
        batch = movie.shape[0]
    y0, y1, x0, x1 = _bbox(mask)
    m = np.asarray(mask, bool)[y0:y1, x0:x1]
    w = None
    if weights is not None:
        w = np.asarray(weights, np.float32)[y0:y1, x0:x1][m]
        w = w / (float(w.sum()) or 1.0)
    lo, hi, step = t.indices(movie.shape[0]) if isinstance(t, slice) else (int(t), int(t) + 1, 1)
    frames = range(lo, hi, step)
    out = np.empty(len(frames), np.float32)
    pos = 0
    for b0 in range(0, len(frames), batch):
        sel = frames[b0 : b0 + batch]
        blk = movie[slice(sel.start, sel.stop, sel.step), y0:y1, x0:x1].reshape(-1, y1 - y0, x1 - x0)
        picked = blk[:, m]
        out[pos : pos + blk.shape[0]] = picked @ w if w is not None else picked.mean(axis=1)
        pos += blk.shape[0]
    return out


def pixel_trace(source, row: int, col: int, t=slice(None), *, z: int = 0, c: int = 0, batch: int = 2000) -> np.ndarray:
    """One pixel's value per frame - ``movie[t, row, col]``.

    Parameters
    ----------
    source
        Anything :func:`as_movie` accepts.
    row, col : int
        Pixel coordinates in the frame.
    t : slice or int, optional
        Frames to read; all by default.
    z, c : int, optional
        Plane and channel passed to :func:`as_movie`.
    batch : int, optional
        Frames per read on a raw movie; a masknmf factorized array is read
        in a single call.

    Returns
    -------
    np.ndarray
        ``(num_frames,)`` float32 trace.
    """
    movie = as_movie(source, z=z, c=c)
    if _factorized(movie.arr):
        batch = movie.shape[0]
    nt, ny, nx = movie.shape
    if not (0 <= row < ny and 0 <= col < nx):
        raise IndexError(f"pixel ({row}, {col}) outside {ny}x{nx}")
    lo, hi, step = t.indices(nt) if isinstance(t, slice) else (int(t), int(t) + 1, 1)
    frames = range(lo, hi, step)
    out = np.empty(len(frames), np.float32)
    pos = 0
    for b0 in range(0, len(frames), batch):
        sel = frames[b0 : b0 + batch]
        blk = movie[slice(sel.start, sel.stop, sel.step), int(row), int(col)].reshape(-1)
        out[pos : pos + blk.size] = blk
        pos += blk.size
    return out


# ---------------------------------------------------------------------------
# registered plane dirs
# ---------------------------------------------------------------------------


def open_registered(plane_dir: str | Path) -> tuple[np.memmap, dict]:
    """``(movie (T, Y, X) int16 memmap, ops)`` for a suite2p-shaped plane dir.

    Prefers the registered ``data.bin``; falls back to ``data_raw.bin`` so an
    unregistered plane still runs (with a warning from the caller).
    """
    plane_dir = Path(plane_dir)
    ops_path = plane_dir / "ops.npy"
    if not ops_path.exists():
        raise FileNotFoundError(f"{plane_dir} has no ops.npy")
    ops = np.load(ops_path, allow_pickle=True).item()
    ly = ops.get("Ly") or ops.get("ly")
    lx = ops.get("Lx") or ops.get("lx")
    if not ly or not lx:
        raise ValueError(f"{ops_path} lacks Ly/Lx")
    ly, lx = int(ly), int(lx)
    for cand in (
        Path(ops.get("reg_file") or ""),
        plane_dir / "data.bin",
        plane_dir / "data_raw.bin",
    ):
        if cand and cand.is_file():
            break
    else:
        raise FileNotFoundError(f"{plane_dir} has neither data.bin nor data_raw.bin")
    nframes = cand.stat().st_size // (ly * lx * 2)
    return np.memmap(cand, dtype=np.int16, mode="r", shape=(nframes, ly, lx)), ops


def plane_index(plane_dir: str | Path, ops: dict | None = None) -> int:
    """0-based z index of a plane dir: ``ops['plane']`` (1-based) or the
    ``zplaneNN`` in the directory name; 0 when neither is present."""
    plane_dir = Path(plane_dir)
    if ops is None and (plane_dir / "ops.npy").exists():
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    p = (ops or {}).get("plane")
    if p is not None:
        return int(p) - 1
    m = _ZPLANE_RE.search(plane_dir.name)
    return int(m.group(1)) - 1 if m else 0


def _find_plane_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if (root / "ops.npy").exists():
        return [root]
    dirs = sorted(
        p.parent
        for p in root.rglob("ops.npy")
        if not p.parent.name.startswith(OUT_PREFIX)
    )
    if not dirs:
        raise FileNotFoundError(f"no ops.npy under {root}")
    return dirs


def plane_store(
    plane_dir: str | Path, fallback: RoiLabelStore | None = None
) -> tuple[RoiLabelStore | None, int]:
    """``(store, z)`` to use for one registered plane dir.

    ROIs drawn on the *registered* movie - ``mbo <plane_dir>`` then the ROI
    tool - are saved as ``<plane_dir>/manual_labels.zarr`` with a single
    plane, so that store wins and ``z`` is ``0``. ROIs drawn on the volume
    root (``mbo <save_path>``) or on the raw data live in ``fallback`` and
    are indexed by the plane dir's own z. Returns ``(None, z)`` when there
    is nothing to draw from.
    """
    plane_dir = Path(plane_dir)
    z_global = plane_index(plane_dir)
    local = plane_dir / SAVE_NAME
    if local.exists():
        st = LabelsZarr.load(local)
        return st, (0 if st.nz == 1 else z_global)
    return fallback, z_global


# ---------------------------------------------------------------------------
# run dirs
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """One run dir's outputs - a ``rois_<tag>/`` subset, a discovery, or a
    full suite2p / masknmf plane - loaded for display.

    ``stat`` coordinates always address the full ``shape`` frame. ``uids``
    maps each row to a persistent store uid (``rois.json``) and
    ``store_indices`` to a store index (legacy ``roi_indices.npy``); either
    is None when absent or when its length disagrees with ``stat`` (masknmf
    may merge or delete seeds).
    """

    path: Path
    kind: str  # "extract" | "demix" | "discover" | "suite2p" | "masknmf"
    z: int
    shape: tuple[int, int]
    stat: np.ndarray
    F: np.ndarray | None
    Fneu: np.ndarray | None
    norm: np.ndarray | None
    iscell: np.ndarray | None
    uids: np.ndarray | None
    store_indices: np.ndarray | None


def load_run_dir(path: str | Path, *, iscell_only: bool = True, logger=None) -> RunResult:
    """Load one output dir (any writer's here, or a plain suite2p plane).

    Parameters
    ----------
    path : str or Path
        Directory holding ``stat.npy`` + ``ops.npy``; ``F.npy`` / ``Fneu.npy``
        / ``norm_traces.npy`` / ``iscell.npy`` are picked up when present.
    iscell_only : bool
        For suite2p detection dirs, keep only rows with ``iscell[:, 0] > 0``;
        the other kinds accept every row already.

    Returns
    -------
    RunResult
        Rows stay in stat order; no crop offsets are applied (every writer
        here stores full-frame coordinates).
    """
    logger = logger or log.get("roi_workflow")
    path = Path(path)
    ops = np.load(path / "ops.npy", allow_pickle=True).item()
    wf = ops.get("roi_workflow") or {}
    kind = wf.get("process") or ("masknmf" if ops.get("pipeline") == "masknmf" else "suite2p")
    if wf.get("plane") is not None:
        z = int(wf["plane"])
    else:
        m = _Z_RE.search(path.name)
        if m:
            z = int(m.group(1)) - 1
        else:
            # vanilla suite2p names planes 0-based: suite2p/plane0, plane1...
            m = re.fullmatch(r"plane(\d+)", path.name)
            z = int(m.group(1)) if m else plane_index(path, ops)
    stat = np.load(path / "stat.npy", allow_pickle=True)

    def _opt(name):
        return np.load(path / name) if (path / name).exists() else None

    F, Fneu, norm = _opt("F.npy"), _opt("Fneu.npy"), _opt("norm_traces.npy")
    iscell = _opt("iscell.npy")
    stale = [n for n, a in (("F.npy", F), ("Fneu.npy", Fneu), ("norm_traces.npy", norm), ("iscell.npy", iscell)) if a is not None and len(a) != len(stat)]
    if stale:
        logger.info(
            f"roi_workflow: {path.name}: {', '.join(stale)} row count disagrees "
            f"with stat.npy ({len(stat)} rois); dropped"
        )
        F = None if "F.npy" in stale else F
        Fneu = None if "Fneu.npy" in stale else Fneu
        norm = None if "norm_traces.npy" in stale else norm
        iscell = None if "iscell.npy" in stale else iscell
    uids = None
    if (path / "rois.json").exists():
        rows = json.loads((path / "rois.json").read_text())
        if len(rows) == len(stat) and all("uid" in r for r in rows):
            uids = np.asarray([int(r["uid"]) for r in rows], np.int64)
        elif len(rows) != len(stat):
            logger.info(
                f"roi_workflow: {path.name}: rois.json lists {len(rows)} seeds "
                f"for {len(stat)} components; per-row ids dropped"
            )
    store_indices = None
    if (path / "roi_indices.npy").exists():
        idx = np.load(path / "roi_indices.npy")
        if len(idx) == len(stat):
            store_indices = np.asarray(idx, np.int64)
        else:
            logger.info(
                f"roi_workflow: {path.name}: roi_indices.npy lists {len(idx)} seeds "
                f"for {len(stat)} components; ignored"
            )
    if iscell_only and kind == "suite2p" and iscell is not None:
        keep = np.asarray(iscell[:, 0] > 0)
        stat = stat[keep]
        F = F[keep] if F is not None else None
        Fneu = Fneu[keep] if Fneu is not None else None
        norm = norm[keep] if norm is not None else None
        uids = uids[keep] if uids is not None else None
        store_indices = store_indices[keep] if store_indices is not None else None
        iscell = iscell[keep]
    return RunResult(
        path=path, kind=str(kind), z=z, shape=(int(ops["Ly"]), int(ops["Lx"])),
        stat=stat, F=F, Fneu=Fneu, norm=norm, iscell=iscell,
        uids=uids, store_indices=store_indices,
    )


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def register(
    input_data,
    save_path: str | Path,
    method: RegisterMethod = "suite2p",
    *,
    planes: list[int] | None = None,
    settings: dict | None = None,
    metadata: dict | None = None,
    frame_indices: list[int] | None = None,
    channel: int | None = None,
    force: bool = False,
    logger=None,
) -> list[Path]:
    """Register ``input_data`` and return the suite2p-shaped plane dirs.

    ``planes`` are 1-based (lsp / masknmf convention); ``None`` means every
    z-plane. ``method="none"`` treats ``input_data`` as already-registered
    plane dir(s) and just lists them.

    - ``suite2p``: ``lbm_suite2p_python.run_volume`` with detection off
      (``roidetect=0``). ``settings`` are extra ops.
    - ``masknmf``: ``mbo_utilities.masknmf.run_plane`` with compression and
      demixing skipped. ``settings`` is a ``MasknmfSettings`` dict; only its
      ``registration`` / ``runtime`` sections are used.
    """
    logger = logger or log.get("roi_workflow")
    save_path = Path(save_path)
    if method == "none":
        return _find_plane_dirs(input_data)

    from mbo_utilities.reader import imread

    arr = input_data if hasattr(input_data, "shape") else imread(input_data)
    nz = _source_nz(arr)
    planes = list(planes) if planes else list(range(1, nz + 1))
    t0 = time.time()

    if method == "suite2p":
        from lbm_suite2p_python import run_volume as lsp_run_volume

        ops = {"do_registration": 1, "roidetect": 0}
        ops.update(settings or {})
        if metadata:
            from mbo_utilities.metadata import get_param

            fs = get_param(metadata, "fs")
            if fs and ops.get("fs") in (None, 10.0):
                ops["fs"] = float(fs)
        logger.info(f"roi_workflow: suite2p registration of planes {planes} -> {save_path}")
        lsp_run_volume(
            arr,
            save_path,
            ops=ops,
            planes=planes,
            force_reg=force,
            replot=False,
            frame_indices=frame_indices,
            reader_kwargs={"channel": channel} if channel is not None else None,
            workers=1,
        )
    elif method == "masknmf":
        from mbo_utilities.masknmf import MasknmfSettings, run_plane as mnmf_run_plane
        from mbo_utilities.masknmf.params import STAGE_FORCE, STAGE_RUN, STAGE_SKIP

        s = MasknmfSettings.from_dict(settings or {})
        s.registration.do_registration = STAGE_FORCE if force else STAGE_RUN
        s.compression.do_compression = STAGE_SKIP
        s.demixing.do_demixing = STAGE_SKIP
        s.runtime.keep_bin = True
        logger.info(f"roi_workflow: masknmf registration of planes {planes} -> {save_path}")
        for p in planes:
            mnmf_run_plane(
                arr,
                save_path,
                plane=p,
                settings=s,
                metadata=metadata,
                frame_indices=frame_indices,
                channel=channel,
                replot=False,
                logger=logger,
            )
    else:
        raise ValueError(f"unknown registration method {method!r}")

    from mbo_utilities.masknmf.runner import generate_plane_dirname

    dirs = []
    for p in planes:
        d = save_path / generate_plane_dirname(p, frame_indices)
        if not (d / "ops.npy").exists():
            # lsp may name planes differently when timepoints are given
            cands = [
                c for c in _find_plane_dirs(save_path) if plane_index(c) == p - 1
            ]
            if not cands:
                raise FileNotFoundError(f"registration produced no plane dir for plane {p}")
            d = cands[0]
        dirs.append(d)
    logger.info(f"roi_workflow: registration done in {time.time() - t0:.1f}s")
    return dirs


# ---------------------------------------------------------------------------
# source bookkeeping shared by extract / demix
# ---------------------------------------------------------------------------


def _source_path(source) -> Path | None:
    if isinstance(source, (str, Path)):
        return Path(source)
    p = getattr(source, "source_path", None) or getattr(source, "fpath", None)
    try:
        return Path(p) if p else None
    except TypeError:
        return None


def _default_out_dir(source, tag: str) -> Path:
    p = _source_path(source)
    if p is None:
        raise ValueError("out_dir is required when the source is an in-memory array")
    base = p if p.is_dir() else p.parent
    return base / f"{OUT_PREFIX}{tag}"


def _movie_fingerprint(movie: PlaneMovie, src: Path | None) -> str:
    """PMD cache key for a movie view; a crop's box keeps its cache separate
    from the full plane's."""
    fp = f"{src}:{movie.shape}:{movie.z}:{movie.c}" if src is not None else f"array:{movie.shape}"
    box = movie.box
    if box is not None:
        fp += ":{}:{}:{}:{}".format(*box)
    return fp


def _reg_file_for(src: Path | None, ops: dict) -> Path | None:
    """The registered binary behind ``src``, when one can be found."""
    parent = (src if src.is_dir() else src.parent) if src is not None else None
    cands = [Path(ops.get("reg_file") or "")]
    if parent is not None:
        cands += [parent / "data.bin", parent / "data_raw.bin"]
    for cand in cands:
        if cand and cand.is_file():
            return cand
    return None


def _ops_for(source, movie: PlaneMovie) -> dict:
    """The plane dir's ``ops.npy`` when there is one, else a minimal ops."""
    p = _source_path(source)
    if p is not None:
        d = p if p.is_dir() else p.parent
        if (d / "ops.npy").exists():
            return np.load(d / "ops.npy", allow_pickle=True).item()
    nt, ly, lx = movie.shape
    ops = {"Ly": ly, "Lx": lx, "nframes": nt, "processing_history": []}
    meta = getattr(movie.arr, "metadata", None) or {}
    try:
        from mbo_utilities.metadata import get_param

        fs = get_param(dict(meta), "fs")
        if fs:
            ops["fs"] = float(fs)
    except Exception:
        pass
    return ops


def _resolve_planes(source, movie_src, store: RoiLabelStore, z: int | None, c: int) -> tuple[int, PlaneMovie]:
    """``(store z, movie)``: the store plane and the matching movie view.

    A single-plane store is plane 0. A single-plane movie (a plane dir, a
    2-D/3-D array) is its only plane whatever the store's z; a volume uses
    the same z as the store. With a multi-plane store and no ``z``, a plane
    dir source supplies it (``zplaneNN`` / ``ops['plane']``).
    """
    if z is None:
        if store.nz == 1:
            z = 0
        elif isinstance(source, (str, Path)) and (Path(source) / "ops.npy").exists():
            z = plane_index(source)
        else:
            raise ValueError(f"store has {store.nz} planes; pass z=")
    z = int(z)
    if not 0 <= z < store.nz:
        raise IndexError(f"z={z} out of range for a {store.nz}-plane store")
    if isinstance(movie_src, PlaneMovie):
        return z, movie_src
    if isinstance(movie_src, (str, Path)):
        from mbo_utilities.reader import imread

        movie_src = imread(movie_src)
    nz = _source_nz(movie_src)
    return z, PlaneMovie(movie_src, z=(z if nz > 1 else 0), c=c)


def _write_subset_outputs(
    out_dir: Path,
    *,
    source,
    ops: dict,
    stat: np.ndarray,
    F: np.ndarray,
    Fneu: np.ndarray,
    kept: list[int],
    store: RoiLabelStore,
    info: dict,
) -> Path:
    from mbo_utilities.masknmf.outputs import merge_ops

    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(F.shape[0])
    np.save(out_dir / "stat.npy", stat)
    np.save(out_dir / "F.npy", F)
    np.save(out_dir / "Fneu.npy", Fneu)
    np.save(out_dir / "spks.npy", np.zeros_like(F))
    np.save(out_dir / "iscell.npy", np.ones((n, 2), np.float32))
    np.save(out_dir / "roi_indices.npy", np.asarray(kept, np.int64))
    names = list(store.label_names)
    records = [
        {
            "index": int(i),
            "uid": int(store.rois[i].uid),
            "z": int(store.rois[i].z),
            "area": int(store.rois[i].area),
            "label": names[store.rois[i].class_index]
            if 0 <= store.rois[i].class_index < len(names)
            else None,
            "note": store.rois[i].note,
        }
        for i in kept
    ]
    (out_dir / "rois.json").write_text(json.dumps(records, indent=1))
    src = _source_path(source)
    reg_file = _reg_file_for(src, ops)
    updates = dict(ops)
    updates.update(
        {
            "save_path": str(out_dir),
            "source": str(src) if src is not None else None,
            "n_rois": n,
            "roi_workflow": info,
            "processing_history": list(ops.get("processing_history") or [])
            + [{"step": f"roi_{info['process']}", **info}],
        }
    )
    if reg_file is not None:
        updates["reg_file"] = str(reg_file)
    merge_ops(out_dir, updates)
    return out_dir


def _write_discovery_outputs(
    out_dir: Path,
    *,
    source,
    ops: dict,
    stat: np.ndarray,
    F: np.ndarray,
    Fneu: np.ndarray,
    info: dict,
) -> Path:
    from mbo_utilities.masknmf.outputs import merge_ops

    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(len(stat))
    np.save(out_dir / "stat.npy", stat)
    np.save(out_dir / "F.npy", F)
    np.save(out_dir / "Fneu.npy", Fneu)
    np.save(out_dir / "spks.npy", np.zeros_like(F))
    np.save(out_dir / "iscell.npy", np.ones((n, 2), np.float32))
    (out_dir / "rois.json").write_text(
        json.dumps([{"z": int(info["plane"]), "npix": int(r["npix"])} for r in stat], indent=1)
    )
    src = _source_path(source)
    reg_file = _reg_file_for(src, ops)
    updates = dict(ops)
    updates.update(
        {
            "save_path": str(out_dir),
            "source": str(src) if src is not None else None,
            "n_rois": n,
            "roi_workflow": info,
            "processing_history": list(ops.get("processing_history") or [])
            + [{"step": f"roi_{info['process']}", **info}],
        }
    )
    if reg_file is not None:
        updates["reg_file"] = str(reg_file)
    merge_ops(out_dir, updates)
    return out_dir


# ---------------------------------------------------------------------------
# extraction (suite2p-style, masks given)
# ---------------------------------------------------------------------------


def _roi_pixels(label_image: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    ypix, xpix = np.nonzero(label_image == k)
    return ypix.astype(np.int32), xpix.astype(np.int32)


def _neuropil_annulus(
    label_image: np.ndarray, k: int, inner: int, outer: int, min_pixels: int
) -> np.ndarray:
    """``(Y, X)`` bool ring around ROI ``k`` excluding every ROI.

    Grows ``outer`` until ``min_pixels`` free pixels are found (or the frame
    is exhausted), like suite2p's ``create_neuropil_masks``.
    """
    from scipy.ndimage import binary_dilation

    cell = label_image == k
    any_cell = label_image > 0
    limit = max(label_image.shape)
    while True:
        inner_zone = binary_dilation(cell, iterations=max(inner, 1))
        outer_zone = binary_dilation(cell, iterations=max(outer, inner + 1))
        ring = outer_zone & ~inner_zone & ~any_cell
        if ring.sum() >= min_pixels or outer >= limit:
            return ring
        outer *= 2


def _sparse_weights(masks: list[np.ndarray], box) -> "scipy.sparse.csr_matrix":  # noqa: F821
    """(K, h*w) CSR averaging each ``(Y, X)`` bool mask inside ``box``."""
    from scipy.sparse import csr_matrix

    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    indptr = np.zeros(len(masks) + 1, np.int64)
    indices, data = [], []
    for k, m in enumerate(masks):
        pix = np.flatnonzero(m[y0:y1, x0:x1])
        indptr[k + 1] = indptr[k] + pix.size
        indices.append(pix)
        data.append(np.full(pix.size, 1.0 / max(pix.size, 1), np.float32))
    if not masks:
        return csr_matrix((0, h * w), dtype=np.float32)
    return csr_matrix(
        (np.concatenate(data), np.concatenate(indices), indptr),
        shape=(len(masks), h * w),
        dtype=np.float32,
    )


def _extract_mean(movie: PlaneMovie, label_image, K, *, neuropil, inner, outer, min_npix, batch):
    """Streaming weighted-mean extraction: (F, Fneu) as (K, T) float32.

    Reads only the union bounding box of the selected ROIs (and their
    neuropil rings), in ``batch``-frame blocks - the cheap path on a
    spatially lazy source, and no worse than full frames on one that isn't.
    """
    T, ly, lx = movie.shape
    cells = [label_image == k for k in range(1, K + 1)]
    rings = (
        [_neuropil_annulus(label_image, k, inner, outer, min_npix) for k in range(1, K + 1)]
        if neuropil
        else []
    )
    union = np.zeros((ly, lx), bool)
    for m in cells + rings:
        union |= m
    box = _bbox(union)
    y0, y1, x0, x1 = box
    Wc = _sparse_weights(cells, box)
    Wn = _sparse_weights(rings, box) if neuropil else None
    F = np.zeros((K, T), np.float32)
    Fneu = np.zeros((K, T), np.float32)
    for t0 in range(0, T, batch):
        blk = movie.frames(t0, min(T, t0 + batch), slice(y0, y1), slice(x0, x1))
        flat = blk.reshape(blk.shape[0], -1).astype(np.float32, copy=False)
        F[:, t0 : t0 + flat.shape[0]] = Wc @ flat.T
        if Wn is not None:
            Fneu[:, t0 : t0 + flat.shape[0]] = Wn @ flat.T
    return F, Fneu


def _extract_suite2p(movie: PlaneMovie, label_image, K, ops, *, neuropil, inner, min_npix, batch, device):
    import torch
    from lbm_suite2p_python import masks_to_stat
    from suite2p.extraction.extract import extract_traces
    from suite2p.extraction.masks import create_masks

    ly, lx = label_image.shape
    stat = masks_to_stat(label_image, ops.get("meanImg"))
    cell_masks, neuropil_masks = create_masks(
        list(stat),
        ly,
        lx,
        neuropil_extract=neuropil,
        inner_neuropil_radius=inner,
        min_neuropil_pixels=min_npix,
    )
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    F, Fneu = extract_traces(movie, cell_masks, neuropil_masks, batch_size=batch, device=dev)
    F = np.asarray(F, np.float32)
    Fneu = np.asarray(Fneu, np.float32) if Fneu is not None else np.zeros_like(F)
    return F, Fneu


def extract_rois(
    source,
    store: RoiLabelStore,
    indices: list[int] | None = None,
    *,
    z: int | None = None,
    c: int = 0,
    out_dir: str | Path | None = None,
    engine: ExtractEngine = "mean",
    neuropil: bool = True,
    inner_neuropil_radius: int = 2,
    outer_neuropil_radius: int = 8,
    min_neuropil_pixels: int = 350,
    batch_size: int = 500,
    device: str = "auto",
    tag: str = "manual",
    logger=None,
) -> Path | None:
    """Extract traces for ``indices`` (store indices; all when None) from ``source``.

    ``source`` is any lazy array, numpy array, ``PlaneMovie``, or a path
    ``imread`` opens (a registered plane dir being the usual case). ``z`` is
    the store plane (0 for a single-plane store); a single-plane source is
    read as is, a volume at that same z. ``c`` picks the channel.

    Returns the output dir (``out_dir`` or ``rois_<tag>/`` beside the
    source) or ``None`` when no selected ROI lives on this plane.

    ``engine="mean"`` is a pure numpy/scipy weighted mean with a dilated-ring
    neuropil - always available. ``engine="suite2p"`` uses suite2p's own
    mask builder and GPU extractor (requires suite2p + lbm_suite2p_python).
    """
    logger = logger or log.get("roi_workflow")
    if indices is None:
        indices = list(range(len(store.rois)))
    z, movie = _resolve_planes(source, source, store, z, c)
    label_image, kept = plane_masks(store, z, indices)
    if not kept:
        logger.info(f"roi_workflow: no selected ROIs on plane z={z}; skipping")
        return None
    if movie.shape[1:] != label_image.shape:
        raise ValueError(
            f"ROI store is {label_image.shape} but the movie is {movie.shape[1:]}; "
            "draw the ROIs on the same field of view"
        )
    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir(source, tag)
    ops = _ops_for(source, movie)
    K = len(kept)
    t0 = time.time()
    logger.info(f"roi_workflow: extracting {K} ROIs on plane z={z} with engine={engine}")
    if engine == "suite2p":
        F, Fneu = _extract_suite2p(
            movie, label_image, K, ops,
            neuropil=neuropil, inner=inner_neuropil_radius,
            min_npix=min_neuropil_pixels, batch=batch_size, device=device,
        )
    elif engine == "mean":
        F, Fneu = _extract_mean(
            movie, label_image, K,
            neuropil=neuropil, inner=inner_neuropil_radius, outer=outer_neuropil_radius,
            min_npix=min_neuropil_pixels, batch=batch_size,
        )
    else:
        raise ValueError(f"unknown extraction engine {engine!r}")

    from mbo_utilities.masknmf.outputs import roi_stat

    ly, lx = label_image.shape
    stat = np.array(
        [
            roi_stat(
                np.ravel_multi_index(_roi_pixels(label_image, k), (ly, lx)),
                np.ones(int((label_image == k).sum()), np.float32),
                (ly, lx),
                F[k - 1],
            )
            for k in range(1, K + 1)
        ],
        dtype=object,
    )
    info = {
        "process": "extract",
        "engine": engine,
        "neuropil": bool(neuropil),
        "roi_indices": [int(i) for i in kept],
        "plane": int(z),
        "channel": int(c),
        "seconds": round(time.time() - t0, 3),
    }
    out = _write_subset_outputs(
        out_dir, source=source, ops=ops, stat=stat, F=F, Fneu=Fneu,
        kept=kept, store=store, info=info,
    )
    logger.info(f"roi_workflow: wrote {K} traces -> {out} ({info['seconds']}s)")
    return out


# ---------------------------------------------------------------------------
# cropping an existing PMD decomposition
# ---------------------------------------------------------------------------


def pmd_crop(pmd, y0: int, y1: int, x0: int, x1: int):
    """Spatially crop a ``masknmf.PMDArray`` without recompressing.

    Row-selects the sparse spatial basis (and the local projector / trend
    basis when present) and crops the mean / variance images, so the result
    is the parent decomposition restricted to the window - exact, and
    effectively free next to a new PMD run on the crop.

    Parameters
    ----------
    pmd : masknmf.PMDArray
        Parent decomposition of shape ``(T, H, W)``.
    y0, y1, x0, x1 : int
        Crop bounds, ``0 <= y0 < y1 <= H`` and ``0 <= x0 < x1 <= W``.

    Returns
    -------
    masknmf.PMDArray
        Decomposition of shape ``(T, y1 - y0, x1 - x0)`` sharing the
        parent's temporal basis, device, and rescale / trend settings.
    """
    import torch
    from masknmf import PMDArray

    nt, h, w = pmd.shape
    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise IndexError(f"crop ({y0}:{y1}, {x0}:{x1}) outside {h}x{w}")
    idx = torch.arange(h * w, device=pmd.device).reshape(h, w)[y0:y1, x0:x1].reshape(-1)
    proj = pmd.u_local_projector
    trend = pmd.spatial_trend_basis
    return PMDArray.from_tensors(
        (nt, y1 - y0, x1 - x0),
        torch.index_select(pmd.u, 0, idx),
        pmd.v,
        pmd.mean_img[y0:y1, x0:x1],
        pmd.var_img[y0:y1, x0:x1],
        u_local_projector=torch.index_select(proj, 0, idx) if proj is not None else None,
        spatial_trend_basis=trend[idx] if trend is not None else None,
        temporal_trend_basis=pmd.temporal_trend_basis if trend is not None else None,
        device=pmd.device,
        rescale=pmd.rescale,
        include_trend=pmd.include_trend,
    )


def _cached_pmd_crop(source, movie: PlaneMovie, cfg, logger) -> tuple[object, str] | None:
    """Cropped ``PMDArray`` built from the source plane's cached compression.

    Parameters
    ----------
    source
        The plane source ``movie`` was opened from; its plane dir is where
        the cached ``compression.hdf5`` is looked up.
    movie : PlaneMovie
        The crop to serve; its ``box`` gives the window.
    cfg : MasknmfCompressionSettings
        Current compression settings; the cache is only reused when its
        stored settings hash matches and compression is not forced.
    logger
        Workflow logger.

    Returns
    -------
    tuple of (masknmf.PMDArray, str) or None
        The cropped decomposition and its provenance key, or None when there
        is no usable cache (no plane dir, no file, stale settings, a shape
        mismatch, or ``movie`` is not a crop).
    """
    from mbo_utilities.masknmf import runner as _runner
    from mbo_utilities.masknmf.params import PMD_FILE, STAGE_FORCE

    box = movie.box
    if box is None or cfg.do_compression == STAGE_FORCE:
        return None
    src = _source_path(source)
    if src is None:
        return None
    pmd_path = (src if src.is_dir() else src.parent) / PMD_FILE
    if not pmd_path.exists():
        return None
    stored = _runner._read_provenance(pmd_path)
    if stored is None or stored.get("settings") != _runner._stage_hash(cfg, "do_compression"):
        return None

    import masknmf

    try:
        pmd = masknmf.PMDArray.from_hdf5(str(pmd_path))
    except Exception as e:
        logger.warning(f"roi_workflow: cached {pmd_path.name} unusable ({e}); recompressing crop")
        return None
    size = dict(zip(movie.dims, (int(s) for s in movie.arr.shape)))
    if tuple(pmd.shape) != (movie.shape[0], size["Y"], size["X"]):
        return None
    y0, y1, x0, x1 = box
    logger.info(f"roi_workflow: cropping cached {pmd_path.name} to ({y0}:{y1}, {x0}:{x1})")
    return pmd_crop(pmd, y0, y1, x0, x1), f"pmd_crop:{pmd_path}:{box}"


# ---------------------------------------------------------------------------
# demixing (masknmf, seeded with the drawn masks)
# ---------------------------------------------------------------------------


def demix_rois(
    source,
    store: RoiLabelStore,
    indices: list[int] | None = None,
    *,
    z: int | None = None,
    c: int = 0,
    out_dir: str | Path | None = None,
    settings: dict | None = None,
    device: str = "auto",
    tag: str = "manual",
    logger=None,
) -> Path | None:
    """masknmf demixing of ``source`` initialised from the drawn ROI masks.

    ``source`` / ``z`` / ``c`` / ``out_dir`` as in :func:`extract_rois`.
    PMD compression runs through the same ``PlaneMovie`` view, so any
    spatially sliceable array works; its result is cached as
    ``compression.hdf5`` next to the outputs (a plane dir's earlier masknmf
    cache is reused). A cropped view of an already-compressed plane skips
    compression entirely: the plane's cached decomposition is cropped from
    its factors (:func:`pmd_crop`). Outputs are masknmf's usual suite2p-shaped sidecars
    plus ``demixing_results.hdf5``.

    NMF may merge or delete seeds, so the number of output components can be
    smaller than the number of seeded ROIs; ``roi_indices.npy`` records the
    seed order and the ``demixing_results.hdf5`` holds the final footprints.
    """
    logger = logger or log.get("roi_workflow")

    import masknmf

    from mbo_utilities.masknmf import MasknmfSettings
    from mbo_utilities.masknmf import outputs as _outputs
    from mbo_utilities.masknmf import runner as _runner
    from mbo_utilities.metadata import get_param

    s = settings if isinstance(settings, MasknmfSettings) else MasknmfSettings.from_dict(settings or {})
    if indices is None:
        indices = list(range(len(store.rois)))
    z, movie = _resolve_planes(source, source, store, z, c)
    label_image, kept = plane_masks(store, z, indices)
    if not kept:
        logger.info(f"roi_workflow: no selected ROIs on plane z={z}; skipping")
        return None
    nframes, ly, lx = movie.shape
    if (ly, lx) != label_image.shape:
        raise ValueError(f"ROI store is {label_image.shape} but the movie is {(ly, lx)}")
    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir(source, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    # a full plane shares the plane's PMD cache; a crop keeps its own
    cache_dir = out_dir if movie.box is not None else out_dir.parent
    ops = _ops_for(source, movie)
    K = len(kept)
    dev = _runner._resolve_device(s.runtime.device if device == "auto" else device, logger)
    fs = get_param(ops, "fs")
    fs = float(fs) if fs else None
    t0 = time.time()
    # masknmf's spline detrenders need a movie longer than their window
    # (40 s for compression, 20 s for demixing); shorter movies skip it
    detrend_ok = bool(fs) and nframes >= 2 * int(40 * fs)
    if fs and not detrend_ok:
        logger.info(
            f"roi_workflow: {nframes} frames < 2x detrend window at fs={fs}; detrending disabled"
        )
        s.compression.detrend = False

    # PMD through the movie view: crop a cached plane decomposition, else
    # reuse this view's cache, else compute
    src = _source_path(source)
    fingerprint = _movie_fingerprint(movie, src)
    cached = _cached_pmd_crop(source, movie, s.compression, logger)
    if cached is not None:
        pmd, comp_seconds, pmd_key = cached[0], 0.0, cached[1]
    else:
        pmd, comp_seconds, pmd_key = _runner._stage_compression(
            movie, s.compression, s.runtime, cache_dir, dev, np.ones((ly, lx), float), fs, logger,
            f"registered:{fingerprint}", False,
        )
        if pmd is None:
            raise ValueError("compression skipped and no cached compression.hdf5 to demix from")

    # seed footprints: one binary column per selected ROI
    a0 = np.zeros((ly, lx, K), np.float32)
    for k in range(1, K + 1):
        a0[..., k - 1] = (label_image == k)

    detrender = None
    if detrend_ok:
        from masknmf.compression.preprocessing import MaximinSplineDetrend

        detrender = MaximinSplineDetrend(
            num_frames=nframes,
            num_knots=max(4, int(nframes / fs / 20)),
            window=int(20 * fs),
            sigma=max(2.0, 0.3 * fs),
            device=dev,
        )
    cfg = _runner.clamp_background_downsampling(s.demixing, ly, lx, logger)
    logger.info(f"roi_workflow: masknmf demixing seeded with {K} ROIs on plane z={z} ({dev})")
    demixer = masknmf.SignalDemixer(pmd, device=dev, frame_batch_size=s.runtime.frame_batch_size)
    demixer.initialize_signals(is_custom=True, spatial_footprints=a0, c_nonneg=True)
    demixer.demix(**cfg.nmf_kwargs(cfg.unfiltered_support_lo, ring=True, detrender=detrender))
    results = demixer.results

    info = {
        "process": "demix",
        "engine": "masknmf",
        "roi_indices": [int(i) for i in kept],
        "plane": int(z),
        "channel": int(c),
        "settings": _runner._stage_hash(cfg, "do_demixing"),
        "input": pmd_key,
        "fs": fs,
    }
    _runner._export_atomic(results, out_dir / _runner.DEMIX_FILE, info)
    coo_idx, values, baseline = _runner._extract_footprints(results)
    cc = np.asarray(results.ac_array.export_c(), dtype=np.float32)
    counts = _outputs.write_plane_outputs(
        out_dir,
        indices=coo_idx, values=values, c=cc, shape=(ly, lx), baseline=baseline,
        var_img=_runner._to_np(getattr(pmd, "var_img", None)),
        mean_img=_runner._to_np(getattr(pmd, "mean_img", None)),
    )
    info["seconds"] = round(time.time() - t0, 3)
    info["compression_seconds"] = round(comp_seconds, 3)
    info["n_components"] = int(counts["n_rois"])
    np.save(out_dir / "roi_indices.npy", np.asarray(kept, np.int64))
    names = list(store.label_names)
    (out_dir / "rois.json").write_text(json.dumps(
        [
            {
                "index": int(i), "uid": int(store.rois[i].uid),
                "z": int(store.rois[i].z), "area": int(store.rois[i].area),
                "label": names[store.rois[i].class_index] if 0 <= store.rois[i].class_index < len(names) else None,
                "note": store.rois[i].note,
            }
            for i in kept
        ],
        indent=1,
    ))
    reg_file = _reg_file_for(src, ops)
    updates = dict(ops)
    updates.update(
        {
            "save_path": str(out_dir),
            "source": str(src) if src is not None else None,
            "n_rois": counts["n_rois"],
            "pipeline": "masknmf",
            "roi_workflow": info,
            "processing_history": list(ops.get("processing_history") or []) + [{"step": "roi_demix", **info}],
        }
    )
    if reg_file is not None:
        updates["reg_file"] = str(reg_file)
    _outputs.merge_ops(out_dir, updates)
    if counts["n_rois"] != K:
        logger.warning(
            f"roi_workflow: seeded {K} ROIs, masknmf kept {counts['n_rois']} components "
            "(merges/deletions); see demixing_results.hdf5"
        )
    logger.info(f"roi_workflow: demixed -> {out_dir} ({info['seconds']}s)")
    return out_dir


# ---------------------------------------------------------------------------
# discovery (unseeded, inside a region)
# ---------------------------------------------------------------------------


def _shift_stat(out_dir: Path, y0: int, x0: int, shape: tuple[int, int]) -> np.ndarray:
    """Move a crop-space ``stat.npy`` into the full ``(Ly, Lx)`` frame."""
    ly, lx = shape
    stat = np.load(Path(out_dir) / "stat.npy", allow_pickle=True)
    for r in stat:
        r["ypix"] = np.clip(r["ypix"] + y0, 0, ly - 1).astype(np.int32)
        r["xpix"] = np.clip(r["xpix"] + x0, 0, lx - 1).astype(np.int32)
        r["med"] = (float(r["med"][0]) + y0, float(r["med"][1]) + x0)
    np.save(Path(out_dir) / "stat.npy", stat)
    return stat


def discover_rois(
    source,
    box: tuple[int, int, int, int],
    *,
    engine: DiscoverEngine = "masknmf",
    z: int = 0,
    c: int = 0,
    out_dir: str | Path | None = None,
    settings: dict | None = None,
    device: str = "auto",
    tag: str = "find",
    logger=None,
) -> Path | None:
    """Detect ROIs inside a rectangular region of ``source``, unseeded.

    ``box`` is ``(y0, y1, x0, x1)`` in frame pixels, clipped to the frame.
    Everything runs on the crop - masknmf's superpixel initialisation
    (``engine="masknmf"``) or suite2p's detector plus its extractor
    (``engine="suite2p"``) - and ``stat.npy`` is written back in full-frame
    coordinates, so the outputs read like any other run dir. When the plane
    already has a ``compression.hdf5`` computed with the same settings, the
    crop's PMD is built from those factors (:func:`pmd_crop`) instead of
    recompressing.

    Returns the output dir (``out_dir`` or ``rois_<tag>/`` beside the
    source), or ``None`` when nothing is found in the region - an ordinary
    outcome, not an error. ``settings`` is a ``MasknmfSettings`` dict for
    masknmf, extra detection settings for suite2p.
    """
    logger = logger or log.get("roi_workflow")
    if engine not in ("masknmf", "suite2p"):
        raise ValueError(f"unknown discovery engine {engine!r}")
    z = int(z)
    if isinstance(source, PlaneMovie):
        movie = source
    else:
        arr = source
        if isinstance(arr, (str, Path)):
            from mbo_utilities.reader import imread

            arr = imread(arr)
        nz = _source_nz(arr)
        movie = PlaneMovie(arr, z=(z if nz > 1 else 0), c=c)
    nframes, ly, lx = movie.shape
    y0, y1, x0, x1 = (int(v) for v in box)
    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, ly), min(x1, lx)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"empty region ({y0}:{y1}, {x0}:{x1}) in a {ly}x{lx} frame")
    crop = movie.crop(y0, y1, x0, x1)
    h, w = y1 - y0, x1 - x0
    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir(source, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = _ops_for(source, movie)
    t0 = time.time()
    info = {
        "process": "discover",
        "engine": engine,
        "box": [y0, y1, x0, x1],
        "plane": movie.z,
        "channel": int(c),
    }

    if engine == "masknmf":
        import masknmf
        from masknmf.demixing import NoSignalsDetectedError

        from mbo_utilities.masknmf import MasknmfSettings
        from mbo_utilities.masknmf import outputs as _outputs
        from mbo_utilities.masknmf import runner as _runner
        from mbo_utilities.metadata import get_param

        s = settings if isinstance(settings, MasknmfSettings) else MasknmfSettings.from_dict(settings or {})
        dev = _runner._resolve_device(s.runtime.device if device == "auto" else device, logger)
        fs = get_param(ops, "fs")
        fs = float(fs) if fs else None
        # masknmf's spline detrenders need a movie longer than their window
        detrend_ok = bool(fs) and nframes >= 2 * int(40 * fs)
        if fs and not detrend_ok:
            s.compression.detrend = False
        cached = _cached_pmd_crop(source, crop, s.compression, logger)
        if cached is not None:
            pmd, comp_seconds, pmd_key = cached[0], 0.0, cached[1]
        else:
            pmd, comp_seconds, pmd_key = _runner._stage_compression(
                crop, s.compression, s.runtime, out_dir, dev, np.ones((h, w), float), fs, logger,
                f"registered:{_movie_fingerprint(crop, _source_path(source))}", False,
            )
            if pmd is None:
                raise ValueError("compression skipped and no cached compression.hdf5 to demix from")
        detrender = None
        if detrend_ok:
            from masknmf.compression.preprocessing import MaximinSplineDetrend

            detrender = MaximinSplineDetrend(
                num_frames=nframes,
                num_knots=max(4, int(nframes / fs / 20)),
                window=int(20 * fs),
                sigma=max(2.0, 0.3 * fs),
                device=dev,
            )
        cfg = _runner.clamp_background_downsampling(s.demixing, h, w, logger)
        logger.info(
            f"roi_workflow: masknmf discovery in ({y0}:{y1}, {x0}:{x1}) on plane z={z} ({dev})"
        )
        demixer = masknmf.SignalDemixer(pmd, device=dev, frame_batch_size=s.runtime.frame_batch_size)
        try:
            demixer.initialize_signals(**cfg.init_kwargs(detrender))
        except NoSignalsDetectedError:
            logger.info("roi_workflow: masknmf found no signals in the region")
            return None
        demixer.demix(**cfg.nmf_kwargs(cfg.unfiltered_support_lo, ring=True, detrender=detrender))
        results = demixer.results
        info.update(settings=_runner._stage_hash(cfg, "do_demixing"), input=pmd_key, fs=fs)
        _runner._export_atomic(results, out_dir / _runner.DEMIX_FILE, info)
        coo_idx, values, baseline = _runner._extract_footprints(results)
        cc = np.asarray(results.ac_array.export_c(), dtype=np.float32)
        counts = _outputs.write_plane_outputs(
            out_dir,
            indices=coo_idx, values=values, c=cc, shape=(h, w), baseline=baseline,
            var_img=_runner._to_np(getattr(pmd, "var_img", None)),
            mean_img=_runner._to_np(getattr(pmd, "mean_img", None)),
        )
        stat = _shift_stat(out_dir, y0, x0, (ly, lx))
        F = np.load(out_dir / "F.npy")
        Fneu = np.load(out_dir / "Fneu.npy")
        info["compression_seconds"] = round(comp_seconds, 3)
        info["n_components"] = int(counts["n_rois"])
    else:
        import torch
        from suite2p import default_settings, detection_wrapper
        from suite2p.extraction.extract import extract_traces
        from suite2p.extraction.masks import create_masks

        from mbo_utilities.metadata import get_param

        dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        det = dict(default_settings()["detection"])
        det.update(settings or {})
        fs = get_param(ops, "fs")
        logger.info(
            f"roi_workflow: suite2p discovery in ({y0}:{y1}, {x0}:{x1}) on plane z={z} ({dev})"
        )
        try:
            _, stat, _ = detection_wrapper(
                crop, fs=float(fs) if fs else 30, yrange=None, xrange=None,
                settings=det, device=dev,
            )
        except ValueError:
            logger.info("roi_workflow: suite2p found no ROIs in the region")
            return None
        cell_masks, neuropil_masks = create_masks(list(stat), h, w)
        F, Fneu = extract_traces(crop, cell_masks, neuropil_masks, device=dev)
        F = np.asarray(F, np.float32)
        Fneu = np.asarray(Fneu, np.float32) if Fneu is not None else np.zeros_like(F)
        for r in stat:
            r["ypix"] = np.clip(r["ypix"] + y0, 0, ly - 1).astype(np.int32)
            r["xpix"] = np.clip(r["xpix"] + x0, 0, lx - 1).astype(np.int32)
            r["med"] = (float(r["med"][0]) + y0, float(r["med"][1]) + x0)

    info["seconds"] = round(time.time() - t0, 3)
    out = _write_discovery_outputs(
        out_dir, source=source, ops=ops, stat=stat, F=F, Fneu=Fneu, info=info,
    )
    logger.info(f"roi_workflow: discovered {len(stat)} ROIs -> {out} ({info['seconds']}s)")
    return out


# ---------------------------------------------------------------------------
# the whole chain
# ---------------------------------------------------------------------------


def run(
    input_data,
    save_path: str | Path | None = None,
    *,
    register_method: RegisterMethod = "suite2p",
    process: ProcessMethod = "extract",
    rois=None,
    selection: RoiSelection | dict | None = None,
    planes: list[int] | None = None,
    register_settings: dict | None = None,
    process_settings: dict | None = None,
    metadata: dict | None = None,
    frame_indices: list[int] | None = None,
    channel: int | None = None,
    force: bool = False,
    tag: str = "manual",
    logger=None,
) -> dict[int, Path]:
    """Register, pick ROIs, and extract or demix them. Returns ``{z: out_dir}``.

    Parameters
    ----------
    input_data
        Raw data (anything ``imread`` opens) - or, with
        ``register_method="none"``, a registered plane dir / a directory of them.
    save_path
        Where registration writes plane dirs. Required unless ``register_method="none"``.
    register_method
        ``"suite2p"``, ``"masknmf"`` or ``"none"``.
    process
        ``"extract"`` (suite2p-style traces from the masks), ``"demix"``
        (masknmf seeded NMF), ``"discover"`` (unseeded detection inside
        ``process_settings["box"]``, engine from ``process_settings["engine"]``,
        no ROI store needed), or ``"none"`` to stop after registration so
        ROIs can be drawn on the registered movie first.
    rois
        ``RoiLabelStore``, labels zarr path, or ``None`` to look for
        ``manual_labels.zarr`` in each plane dir (drawn on the registered
        movie - preferred), then beside ``input_data``, then in ``save_path``.
    selection
        ``RoiSelection`` (or its dict) restricting which ROIs run; ``None`` = all.
    planes
        1-based z-planes to register/process; ``None`` = all. Independent of
        ``selection.planes`` (0-based) - a plane with no selected ROI is skipped.
    register_settings / process_settings
        Passed through to ``register`` and ``extract_rois`` / ``demix_rois``.
    """
    logger = logger or log.get("roi_workflow")
    if register_method != "none" and save_path is None:
        raise ValueError("save_path is required when registering")
    if process not in ("extract", "demix", "discover", "none"):
        raise ValueError(f"unknown process {process!r}")
    if process == "discover" and not (process_settings or {}).get("box"):
        raise ValueError('process="discover" needs process_settings["box"] = [y0, y1, x0, x1]')
    source_path = None if hasattr(input_data, "shape") else input_data
    sel = selection if isinstance(selection, RoiSelection) else RoiSelection(**(selection or {}))

    plane_dirs = register(
        input_data, save_path or "", register_method,
        planes=planes, settings=register_settings, metadata=metadata,
        frame_indices=frame_indices, channel=channel, force=force, logger=logger,
    )
    if planes and register_method == "none":
        plane_dirs = [d for d in plane_dirs if plane_index(d) + 1 in set(planes)]

    if process == "none":
        # register only: draw ROIs on the registered movie next
        root = Path(save_path) if save_path else Path(plane_dirs[0]).parent
        logger.info(
            f"roi_workflow: registered {len(plane_dirs)} plane(s); draw ROIs on them with "
            f"`mbo {root}` (or `mbo <plane_dir>`) then run with --register none"
        )
        return {plane_index(d): d for d in plane_dirs}

    if process == "discover":
        ps = dict(process_settings or {})
        box = tuple(int(v) for v in ps.pop("box"))
        engine = ps.pop("engine", "masknmf")
        found: dict[int, Path] = {}
        for d in plane_dirs:
            z_global = plane_index(d)
            if sel.planes and z_global not in set(sel.planes):
                continue
            out = discover_rois(
                d, box, engine=engine, z=z_global, c=channel or 0,
                tag=tag, logger=logger, **ps,
            )
            if out is not None:
                found[z_global] = out
        if not found:
            logger.warning("roi_workflow: discovery found nothing; nothing written")
        return found

    # a shared store: given explicitly, or saved beside the raw input / root
    shared: RoiLabelStore | None = None
    if rois is not None:
        shared = load_rois(rois)
    elif source_path is not None and labels_path(source_path).exists():
        shared = load_rois(None, source=source_path)
    elif save_path and (Path(save_path) / SAVE_NAME).exists():
        shared = LabelsZarr.load(Path(save_path) / SAVE_NAME)

    outputs: dict[int, Path] = {}
    n_selected = 0
    n_stores = 0
    for d in plane_dirs:
        store, z = plane_store(d, shared)
        z_global = plane_index(d)
        if store is None:
            logger.warning(f"roi_workflow: no ROI store for {d.name}; draw ROIs with `mbo {d}`")
            continue
        n_stores += 1
        if sel.planes and z_global not in set(sel.planes):
            continue
        # per-plane stores are single-plane: the plane filter was applied on
        # the global index above and must not be passed to select_rois
        indices = select_rois(
            store,
            planes=sel.planes if store is shared else None,
            indices=sel.indices or None,
            labels=sel.labels or None,
        )
        n_selected += len(indices)
        if not indices:
            continue
        # each plane dir opens as a single-plane movie; ``z`` addresses the store
        kw = dict(z=z, c=channel or 0, tag=tag, logger=logger)
        if process == "extract":
            out = extract_rois(d, store, indices, **kw, **(process_settings or {}))
        else:
            out = demix_rois(d, store, indices, **kw, settings=process_settings)
        if out is not None:
            outputs[z_global] = out
    if n_stores == 0:
        raise FileNotFoundError(
            "no ROIs found: pass rois=, or draw them on the registered planes "
            f"(`mbo {save_path or plane_dirs[0]}`) or beside the raw data"
        )
    if n_selected == 0:
        raise ValueError("ROI selection matched no ROIs")
    if not outputs:
        logger.warning("roi_workflow: no plane had a selected ROI; nothing written")
    return outputs
