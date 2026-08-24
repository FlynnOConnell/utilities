"""Convert masknmf demixing results into suite2p-shaped plane outputs.

Writing ``stat.npy`` / ``iscell.npy`` / ``F.npy`` / ``Fneu.npy`` / ``spks.npy``
/ ``ops.npy`` into each plane dir makes the whole downstream ecosystem work
unchanged: LBM-Suite2p-Python's QC figure suite, the GUI diagnostics and
summary-image widgets, ``Suite2pArray`` loading, and ``mbo info`` results
detection are all filename-coupled, not class-coupled.

Everything here is numpy-only; the runner extracts tensors from masknmf
objects before calling in.
"""

from pathlib import Path

import numpy as np


def split_sparse_footprints(
    indices: np.ndarray, values: np.ndarray, n_rois: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split COO (pixel, roi) indices/values into per-ROI (pixel_idx, lam).

    ``indices`` is (2, nnz) with row 0 = flat pixel index (C-order over H,W)
    and row 1 = roi index; ``values`` is (nnz,).
    """
    order = np.argsort(indices[1], kind="stable")
    pix = indices[0][order]
    roi = indices[1][order]
    lam = values[order]
    bounds = np.searchsorted(roi, np.arange(n_rois + 1))
    return [
        (pix[bounds[k]: bounds[k + 1]], lam[bounds[k]: bounds[k + 1]])
        for k in range(n_rois)
    ]


def roi_stat(
    pixel_idx: np.ndarray,
    lam: np.ndarray,
    shape: tuple[int, int],
    trace: np.ndarray | None = None,
) -> dict:
    """Build one suite2p-style stat dict from a flat-indexed weighted mask."""
    ypix, xpix = np.unravel_index(pixel_idx.astype(np.int64), shape)
    npix = int(ypix.size)
    if npix == 0:
        med = (0.0, 0.0)
        radius = 0.0
        compact = 0.0
    else:
        med = (float(np.median(ypix)), float(np.median(xpix)))
        radius = float(np.sqrt(npix / np.pi))
        dists = np.sqrt((ypix - med[0]) ** 2 + (xpix - med[1]) ** 2)
        # mean distance from center normalized by the equivalent-disc mean
        # distance (2/3 r); ~1 for a filled disc, grows for stragglier masks
        compact = float(np.mean(dists) / max(radius * (2.0 / 3.0), 1e-6))
    skew = 0.0
    if trace is not None and trace.size > 2:
        t = trace.astype(np.float64)
        sd = t.std()
        if sd > 0:
            skew = float(np.mean(((t - t.mean()) / sd) ** 3))
    return {
        "ypix": ypix.astype(np.int32),
        "xpix": xpix.astype(np.int32),
        "lam": lam.astype(np.float32),
        "med": med,
        "npix": npix,
        "radius": radius,
        "compact": compact,
        "skew": skew,
        "overlap": np.zeros(npix, dtype=bool),
    }


def roi_baselines(
    footprints: list[tuple[np.ndarray, np.ndarray]], baseline: np.ndarray | None
) -> np.ndarray:
    """Per-ROI static-background level: lam-weighted mean of ``baseline``.

    Demixed temporal components are baseline-subtracted; adding this back
    gives F traces with a realistic F0 so percentile dF/F stays finite.
    """
    n = len(footprints)
    out = np.zeros(n, dtype=np.float32)
    if baseline is None:
        return out
    flat = np.asarray(baseline, dtype=np.float32).ravel()
    for k, (pix, lam) in enumerate(footprints):
        if pix.size == 0:
            continue
        w = lam.astype(np.float64)
        s = w.sum()
        if s > 0:
            out[k] = float((flat[pix.astype(np.int64)] * w).sum() / s)
    return out


def write_plane_outputs(
    plane_dir: str | Path,
    *,
    indices: np.ndarray,
    values: np.ndarray,
    c: np.ndarray,
    shape: tuple[int, int],
    baseline: np.ndarray | None = None,
) -> dict:
    """Write stat/iscell/F/Fneu/spks/norm_traces for one plane.

    ``c`` is (T, K) demixed traces; ``indices``/``values`` the COO spatial
    footprints. Returns summary counts for logging/ops.
    """
    plane_dir = Path(plane_dir)
    n_rois = int(c.shape[1]) if c.ndim == 2 else 0
    footprints = split_sparse_footprints(indices, values, n_rois)

    F = np.ascontiguousarray(c.T, dtype=np.float32)
    F += roi_baselines(footprints, baseline)[:, None]

    stat = np.array(
        [
            roi_stat(pix, lam, shape, F[k])
            for k, (pix, lam) in enumerate(footprints)
        ],
        dtype=object,
    )
    # masknmf has no accept/reject classifier: everything demixed is accepted
    iscell = np.ones((n_rois, 2), dtype=np.float32)

    mu = F.mean(axis=1, keepdims=True)
    sd = F.std(axis=1, keepdims=True)
    norm = (F - mu) / np.maximum(sd, 1e-6)

    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "iscell.npy", iscell)
    np.save(plane_dir / "F.npy", F)
    np.save(plane_dir / "Fneu.npy", np.zeros_like(F))
    np.save(plane_dir / "spks.npy", np.zeros_like(F))
    np.save(plane_dir / "norm_traces.npy", norm.astype(np.float32))
    return {"n_rois": n_rois}


def merge_ops(plane_dir: str | Path, updates: dict) -> dict:
    """Merge ``updates`` into the plane dir's ops.npy (create if missing)."""
    ops_path = Path(plane_dir) / "ops.npy"
    ops: dict = {}
    if ops_path.exists():
        loaded = np.load(ops_path, allow_pickle=True)
        try:
            ops = dict(loaded.item())
        except (ValueError, AttributeError):
            ops = {}
    ops.update(updates)
    ops["save_path"] = str(plane_dir)
    ops["ops_path"] = str(ops_path)
    np.save(ops_path, ops)
    return ops
