"""Run orchestration and derived-set state for the manual ROI widget.

``RoiRunManager`` tracks the widget's background work: in-process runs
(extract / demix / discover on a thread, shown as process-manager jobs)
and full-plane suite2p / masknmf spawns (detached subprocesses polled
through their sidecars). GPU-heavy runs serialize on one lock so two
demixes never fight over the device.

``DerivedSet`` wraps a loaded :class:`~mbo_utilities.roi_workflow.RunResult`
for display - per-set color, visibility, discarded rows and a pick map -
and ``TraceSet`` holds traces keyed by store uid so deleting an ROI never
remaps anyone else's rows. Everything here is figure-free and imports
without masknmf, so it is unit-testable with a stub process manager.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.annotation import ROI_COLORS, class_color
from mbo_utilities.gui.widgets.process_manager import LocalJob, get_process_manager
from mbo_utilities.roi_workflow import OUT_PREFIX, RunResult, labels_path

__all__ = [
    "DerivedSet",
    "RoiRun",
    "RoiRunManager",
    "SELECTED_ALPHA",
    "SET_COLORS",
    "TraceSet",
    "build_pick_map",
    "component_color",
    "derived_rgba",
    "display_fneu",
    "display_trace",
    "feathered_rgba",
    "full_plane_args",
    "load_run_registry",
    "registry_path",
    "save_run_registry",
    "scan_run_dirs",
    "set_color",
]

REGISTRY_NAME = "roi_runs.json"
SELECTED_ALPHA = 0.9  # matches the manual overlay's selection opacity

# per-set overlay hues, bright against gnuplot2 and the class colors
SET_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.85, 0.10),
    (0.10, 0.90, 0.90),
    (1.00, 0.45, 0.85),
    (0.45, 1.00, 0.35),
    (1.00, 0.55, 0.15),
    (0.55, 0.55, 1.00),
)


def set_color(index: int) -> tuple[float, float, float]:
    """rgb in 0-1 for a derived set (wraps past the palette end)."""
    return SET_COLORS[index % len(SET_COLORS)]


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


@dataclass
class RoiRun:
    """One background run: an in-process job (``job``) or a spawn (``pid``)."""

    kind: str  # "extract" | "demix" | "discover" | "suite2p" | "masknmf"
    tag: str
    description: str
    job: LocalJob | None = None
    pid: int | None = None
    out_root: Path | None = None
    out_dirs: list[Path] = field(default_factory=list)
    planes: list[int] = field(default_factory=list)
    box: tuple[int, int, int, int] | None = None
    error: str | None = None
    finished: bool = False
    loaded: bool = False


def _raised_at(error: BaseException) -> str:
    """`` (file.py:123)`` for the innermost frame of ``error``, or ""."""
    import traceback

    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return ""
    last = frames[-1]
    return f" ({Path(last.filename).name}:{last.lineno})"


class RoiRunManager:
    """Submit, poll and stop the ROI widget's background runs.

    In-process runs go through :meth:`submit` - a ``LocalJob`` in the
    process console plus a daemon thread; ``heavy=True`` runs queue on one
    lock so the GPU is never shared. Full-plane pipelines go through
    :meth:`spawn`. :meth:`poll` (once per frame) yields each finished run
    exactly once.

    Parameters
    ----------
    pm : optional
        Process manager for ``submit`` / ``spawn``; the global one when
        None. Tests inject a stub here and into ``poll``.
    """

    def __init__(self, pm=None):
        self._pm = pm
        self.runs: list[RoiRun] = []
        self._results: queue.Queue = queue.Queue()
        self._heavy = threading.Lock()
        self.logger = log.get("gui.roi_runs")

    def submit(self, run: RoiRun, fn, *, heavy: bool = False) -> RoiRun:
        """Run ``fn(job)`` on a daemon thread; its return value reaches
        :meth:`poll` as the run's payload (None and ``run.error`` set when
        it raised)."""
        pm = self._pm or get_process_manager()
        run.job = pm.start_job(run.kind, run.description)

        def work():
            job = run.job
            try:
                if heavy:
                    job.set_progress(0.0, "waiting for gpu")
                    with self._heavy:
                        payload = fn(job)
                else:
                    payload = fn(job)
            except Exception as error:  # noqa: BLE001 - reported on the job
                self.logger.exception(f"{run.description} failed")
                message = f"{type(error).__name__}: {error}{_raised_at(error)}"
                job.fail(message)
                self._results.put((run, None, message))
                return
            job.done()
            self._results.put((run, payload, None))

        thread = threading.Thread(target=work, name=f"roi-run-{run.tag}", daemon=True)
        thread.start()
        self.runs.append(run)
        return run

    def spawn(self, run: RoiRun, task_type: str, args: dict) -> RoiRun:
        """Start a detached worker for ``run``; progress arrives through the
        process manager's sidecar polling."""
        pm = self._pm or get_process_manager()
        if run.out_root is None and args.get("output_dir"):
            run.out_root = Path(args["output_dir"])
        pid = pm.spawn(
            task_type,
            args,
            run.description,
            output_path=str(run.out_root) if run.out_root is not None else None,
        )
        if pid is None:
            self._results.put((run, None, "failed to start worker"))
        else:
            run.pid = pid
        self.runs.append(run)
        return run

    def poll(self, pm) -> list[tuple[RoiRun, object]]:
        """``(run, payload)`` per newly finished run; errors land on
        ``run.error`` (spawned runs always carry a None payload)."""
        done: list[tuple[RoiRun, object]] = []
        while True:
            try:
                run, payload, error = self._results.get_nowait()
            except queue.Empty:
                break
            run.finished = True
            run.error = error
            done.append((run, payload))
        spawned = [r for r in self.runs if r.pid is not None and not r.finished]
        if spawned:
            by_pid = {p.pid: p for p in pm.get_running()}
            for run in spawned:
                info = by_pid.get(run.pid)
                if info is None:
                    # a completed entry is pruned from the console after a
                    # few minutes; outputs on disk mean the run succeeded
                    if run.out_root is None or not finished_dirs(run.out_root, run.planes):
                        run.error = "process stopped"
                elif info.status == "completed":
                    pass
                elif info.status == "error":
                    run.error = info.status_message or "process failed"
                else:
                    continue
                run.finished = True
                done.append((run, None))
        return done

    def stop(self, run: RoiRun, pm) -> None:
        """Kill a spawned run; in-process runs cannot be cancelled."""
        if run.pid is not None and not run.finished:
            pm.kill(run.pid)

    @property
    def active(self) -> list[RoiRun]:
        return [r for r in self.runs if not r.finished]

    @property
    def busy(self) -> bool:
        return bool(self.active)


# ---------------------------------------------------------------------------
# derived sets and traces
# ---------------------------------------------------------------------------


def build_pick_map(stat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """``(Y, X)`` int32 row index per pixel, -1 background.

    Rows are painted in ascending peak-lam order, so the strongest
    component wins contested pixels - picking matches what the overlay
    shows on top.
    """
    pick = np.full(shape, -1, np.int32)
    peaks = [float(np.max(s["lam"])) if len(s["lam"]) else 0.0 for s in stat]
    for k in np.argsort(peaks, kind="stable"):
        s = stat[k]
        pick[s["ypix"], s["xpix"]] = k
    return pick


@dataclass(eq=False)
class DerivedSet:
    """One loaded run's components, as shown: overlay color, visibility,
    discarded rows, accepted flags, per-row class labels and the pick map.
    Promoted rows are never stored here - the widget recomputes them from
    the store's ``source`` strings.

    ``accepted`` starts from the run's ``iscell`` and is the mutable
    curation state; the widget mirrors changes back into ``iscell.npy``.
    ``classes`` maps a row to a class index in the shared label set;
    ``colors`` maps a row to an explicit group color (float rgb in 0-1)
    that wins over the class / hue color.
    """

    result: RunResult
    name: str
    color: tuple[float, float, float]
    visible: bool = True
    discarded: set[int] = field(default_factory=set)
    classes: dict[int, int] = field(default_factory=dict)
    colors: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    accepted: np.ndarray | None = None
    pick_map: np.ndarray | None = None

    def __post_init__(self):
        if self.pick_map is None:
            self.pick_map = build_pick_map(self.result.stat, self.result.shape)
        if self.accepted is None:
            iscell = self.result.iscell
            if iscell is not None and len(iscell) == len(self.result.stat):
                self.accepted = np.asarray(iscell)[:, 0] > 0
            else:
                self.accepted = np.ones(len(self.result.stat), bool)


@dataclass
class TraceSet:
    """Traces of one origin ("quick" or a run), keyed by store uid so a
    delete only prunes - other ROIs' rows never move."""

    name: str
    kind: str
    data: dict[int, dict] = field(default_factory=dict)
    visible: bool = True

    def prune(self, uids) -> None:
        """Drop entries whose uid is not in ``uids``."""
        keep = {int(u) for u in uids}
        for uid in [u for u in self.data if u not in keep]:
            del self.data[uid]


def _rim(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask (4-connected)."""
    core = mask.copy()
    core[1:, :] &= mask[:-1, :]
    core[:-1, :] &= mask[1:, :]
    core[:, 1:] &= mask[:, :-1]
    core[:, :-1] &= mask[:, 1:]
    return mask & ~core


def derived_rgba(
    shape: tuple[int, int],
    sets_on_z: list[DerivedSet],
    alpha: float,
    selected: tuple[DerivedSet, int] | None = None,
    grouped: frozenset | set = frozenset(),
) -> np.ndarray:
    """``(ny, nx, 4)`` uint8 overlay for the derived sets of one plane.

    Each component's pixels get its own color (``component_color``) at
    ``lam / lam.max() * alpha``; where components overlap the higher alpha
    wins color and coverage. ``selected`` (a ``(set, row)`` pair) is filled
    at ``SELECTED_ALPHA`` with a white rim; ``grouped`` (``(id(set), row)``
    pairs of a multi-selection) fills at ``SELECTED_ALPHA`` too. Discarded
    rows and invisible sets are skipped; rejected rows draw dimmed.
    """
    comps = []
    sel = None
    for s in sets_on_z:
        if not s.visible:
            continue
        for k, row in enumerate(s.result.stat):
            if k in s.discarded:
                continue
            fill = alpha if s.accepted[k] else alpha * 0.35
            if (id(s), k) in grouped:
                fill = SELECTED_ALPHA
            comps.append((row["ypix"], row["xpix"], row["lam"], component_color(s, k), fill))
    if selected is not None:
        s, k = selected
        if s in sets_on_z and s.visible and k not in s.discarded:
            row = s.result.stat[k]
            sel = (row["ypix"], row["xpix"], component_color(s, k))
    return feathered_rgba(shape, comps, sel)


def display_trace(entry: dict, correct_neuropil: bool = True) -> np.ndarray:
    """A trace the way lbm_suite2p_python plots it: the run's norm_traces
    when present, else percent dF/F over a static 20th-percentile baseline,
    neuropil-corrected (``F - 0.7 * Fneu``) unless turned off."""
    if "norm" in entry:
        return np.asarray(entry["norm"], np.float32)
    f = np.asarray(entry["F"], np.float32)
    if correct_neuropil and "Fneu" in entry:
        f = f - 0.7 * np.asarray(entry["Fneu"], np.float32)
    if not f.size or not np.any(f):
        return np.zeros_like(f)
    f0 = max(float(np.percentile(f, 20)), 1e-6)
    return (f - f0) / f0 * 100.0


def display_fneu(entry: dict) -> np.ndarray | None:
    """The neuropil trace on the same percent scale, or None without one."""
    if "Fneu" not in entry:
        return None
    f = np.asarray(entry["Fneu"], np.float32)
    if not f.size or not np.any(f):
        return np.zeros_like(f)
    f0 = max(float(np.percentile(f, 20)), 1e-6)
    return (f - f0) / f0 * 100.0


def component_color(s: DerivedSet, k: int) -> tuple[float, float, float]:
    """One component's rgb in 0-1: its explicit group color when set, its
    class color when labeled, else a hue of its own - the same treatment
    drawn ROIs get."""
    rgb = s.colors.get(k)
    if rgb is not None:
        return tuple(rgb)
    ci = s.classes.get(k)
    if ci is not None:
        return class_color(ci)
    offset = sum(map(ord, s.name)) * 61
    r, g, b = ROI_COLORS[(k + offset) % len(ROI_COLORS)]
    return (r / 255.0, g / 255.0, b / 255.0)


def feathered_rgba(shape: tuple[int, int], comps, selected=None) -> np.ndarray:
    """``(ny, nx, 4)`` uint8 overlay from ``(ypix, xpix, lam, rgb, fill)``
    components: each pixel takes ``lam / lam.max() * fill`` alpha, and
    where components overlap the higher alpha wins color and coverage.
    ``selected`` (``(ypix, xpix, rgb)``) fills at ``SELECTED_ALPHA`` with
    a white rim.
    """
    ny, nx = shape
    rgba = np.zeros((ny, nx, 4), np.uint8)
    best = np.zeros((ny, nx), np.float32)
    for ypix, xpix, lam, rgb, fill in comps:
        color = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        lam = np.asarray(lam, np.float32)
        peak = float(lam.max()) if lam.size else 0.0
        a = lam / peak * fill if peak > 0 else np.full(lam.shape, fill, np.float32)
        win = a > best[ypix, xpix]
        yy, xx = ypix[win], xpix[win]
        best[yy, xx] = a[win]
        rgba[yy, xx, :3] = color
        rgba[yy, xx, 3] = np.rint(a[win] * 255).astype(np.uint8)
    if selected is not None:
        ypix, xpix, rgb = selected
        mask = np.zeros((ny, nx), bool)
        mask[ypix, xpix] = True
        fill = np.uint8(round(SELECTED_ALPHA * 255))
        rgba[mask, :3] = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        rgba[mask, 3] = fill
        rgba[_rim(mask)] = (255, 255, 255, fill)
    return rgba


# ---------------------------------------------------------------------------
# disk: run dirs and the registry sidecar
# ---------------------------------------------------------------------------


def _kind_of(ops: dict) -> str:
    wf = ops.get("roi_workflow") or {}
    return str(
        wf.get("process")
        or ("masknmf" if ops.get("pipeline") == "masknmf" else "suite2p")
    )


def run_dir_complete(d) -> bool:
    """True when ``d`` holds a loadable run (``stat.npy`` + ``ops.npy``)."""
    d = Path(d)
    return (d / "stat.npy").exists() and (d / "ops.npy").exists()


def finished_dirs(out_root, planes=None) -> list[Path]:
    """Completed output dirs under ``out_root`` for 1-based ``planes`` (any
    plane when None): the root itself when it holds outputs, else its
    ``zplaneNN*`` children - pipelines may suffix the name with a frame
    range, so match by prefix."""
    root = Path(out_root)
    if run_dir_complete(root):
        return [root]
    pats = [f"zplane{int(p):02d}*" for p in planes] if planes else ["zplane*"]
    out: list[Path] = []
    for pat in pats:
        out += [d for d in sorted(root.glob(pat)) if d.is_dir() and run_dir_complete(d)]
    return out


def scan_run_dirs(fpath) -> list[dict]:
    """Loadable run dirs beside ``fpath``, newest first.

    Covers ``rois_<tag>/`` dirs (and their per-plane ``zNN/`` children) and
    sibling ``zplane*`` trees from full-plane pipelines (including their own
    ``rois_*`` subsets), vanilla suite2p ``plane*`` / ``suite2p/plane*`` dirs,
    and the opened dir itself when it holds results. Rows are ``{"path", "kind", "n_rois", "mtime"}``; a
    dir without ``stat.npy`` + ``ops.npy`` (still being written, say) is
    not listed.
    """
    base = labels_path(fpath).parent
    if not base.is_dir():
        return []
    dirs: list[Path] = []

    def add(d: Path):
        if (d / "stat.npy").exists() and (d / "ops.npy").exists():
            dirs.append(d)

    add(base)  # the opened data may itself sit in a result dir
    for d in sorted(base.glob(f"{OUT_PREFIX}*")):
        if not d.is_dir():
            continue
        add(d)
        for child in sorted(d.glob("z[0-9]*")):
            if child.is_dir():
                add(child)
    for d in sorted(base.glob("zplane*")):
        if not d.is_dir():
            continue
        add(d)
        for sub in sorted(d.glob(f"{OUT_PREFIX}*")):
            if sub.is_dir():
                add(sub)
    for pattern in ("plane[0-9]*", "suite2p/plane[0-9]*"):
        for d in sorted(base.glob(pattern)):
            if d.is_dir():
                add(d)

    rows = []
    for d in dirs:
        try:
            ops = np.load(d / "ops.npy", allow_pickle=True).item()
        except Exception:
            continue
        n = ops.get("n_rois")
        if n is None:
            try:
                n = len(np.load(d / "stat.npy", allow_pickle=True))
            except Exception:
                n = 0
        rows.append(
            {
                "path": d,
                "kind": _kind_of(ops),
                "n_rois": int(n),
                "mtime": (d / "ops.npy").stat().st_mtime,
            }
        )
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def full_plane_args(kind: str, fpath, plane_1based: int, iw) -> dict:
    """Minimal ``task_suite2p`` / ``task_masknmf`` worker args for one
    plane, writing beside the data; the worker fills in every default."""
    if fpath is None:
        raise ValueError("no data path to run a full plane on")
    if kind not in ("suite2p", "masknmf"):
        raise ValueError(f"unknown pipeline {kind!r}")
    from mbo_utilities.reader import widget_reader_kwargs

    args = {
        "input_path": str(fpath),
        "output_dir": str(labels_path(fpath).parent),
        "planes": [int(plane_1based)],
        "reader_kwargs": widget_reader_kwargs(iw),
    }
    if kind == "masknmf":
        from mbo_utilities.masknmf.params import MasknmfSettings

        args["settings"] = MasknmfSettings().to_dict()
    return args


def registry_path(fpath) -> Path:
    """``roi_runs.json`` beside ``manual_labels.zarr``."""
    return labels_path(fpath).parent / REGISTRY_NAME


def load_run_registry(path) -> list[dict]:
    """The sidecar's run entries; unreadable or absent files are ``[]``.

    Entries are kept whether or not their dir still exists - a spawned run
    may still be writing when the GUI comes back.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    out = []
    for row in data.get("runs", []) if isinstance(data, dict) else []:
        if isinstance(row, dict) and row.get("path"):
            out.append(
                {
                    "path": str(row["path"]),
                    "kind": str(row.get("kind", "")),
                    "discarded": [int(i) for i in row.get("discarded", [])],
                    "classes": {
                        int(k): int(v)
                        for k, v in (row.get("classes") or {}).items()
                    },
                    "colors": {
                        int(k): tuple(float(x) for x in v)
                        for k, v in (row.get("colors") or {}).items()
                    },
                }
            )
    return out


def save_run_registry(path, entries: list[dict]) -> None:
    """Write the sidecar: ``{"runs": [{"path", "kind", "discarded", "classes",
    "colors"}]}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = [
        {
            "path": str(e["path"]),
            "kind": str(e.get("kind", "")),
            "discarded": sorted(int(i) for i in e.get("discarded", [])),
            "classes": {
                str(k): int(v) for k, v in (e.get("classes") or {}).items()
            },
            "colors": {
                str(k): [float(x) for x in v]
                for k, v in (e.get("colors") or {}).items()
            },
        }
        for e in entries
    ]
    path.write_text(json.dumps({"runs": runs}, indent=1))
