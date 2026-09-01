"""Convert masknmf demixing results into suite2p-shaped plane outputs.

Writing ``stat.npy`` / ``iscell.npy`` / ``F.npy`` / ``Fneu.npy`` / ``spks.npy``
/ ``ops.npy`` into each plane dir makes the whole downstream ecosystem work
unchanged: LBM-Suite2p-Python's QC figure suite, the GUI diagnostics and
summary-image widgets, ``Suite2pArray`` loading, and ``mbo info`` results
detection are all filename-coupled, not class-coupled.

Everything here is numpy-only; the runner extracts tensors from masknmf
objects before calling in.

The suite2p *filenames* are matched; the suite2p *semantics* are not, and
where they conflict masknmf wins. ``F.npy`` is calibrated back to movie
units from the PMD standardisation rather than being a raw pixel average,
``norm_traces.npy`` is dF/F against masknmf's static baseline rather than a
z-score or a rolling percentile, and ``Fneu``/``spks`` are zeros because
there is no neuropil trace and no deconvolution. See ``roi_calibration``.
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


def roi_calibration(
    footprints: list[tuple[np.ndarray, np.ndarray]],
    *,
    var_img: np.ndarray | None = None,
    mean_img: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ROI ``(gain, F0)`` that undo the PMD standardisation.

    masknmf demixes ``(movie - mean_img) / var_img`` — and ``var_img`` is the
    per-pixel noise *std*, not a variance — so ``c`` is in noise-SD units.
    Worse, HALS never renormalises ``a`` or ``c``, so only the product
    ``a_k c_k`` is identified: the amplitude split between the two factors is
    whatever initialisation handed the solver. Raw ``c`` amplitudes are
    therefore not comparable across ROIs and carry no physical unit.

    Calibration fixes that gauge by moving the amplitude into ``a``'s physical
    units and reading F0 off masknmf's own static baseline ``b`` (the term the
    demixer already factored out of the traces)::

        gain_k = mean_support(a_k * var_img)
        F0_k   = mean_support(b * var_img + mean_img)
        F_k    = c_k * gain_k + F0_k              # movie units
        dFF_k  = c_k * gain_k / F0_k              # unitless

    This is masknmf's own ``produce_baseline_corrected_outputs`` convention:
    an *unweighted* mean over each footprint's support, not a lam-weighted
    one. Keeping it identical means traces here and traces from a direct
    masknmf script agree numerically.

    Missing inputs degrade gracefully: no ``var_img`` leaves ``gain`` as the
    mean lam (standardised units), and no ``mean_img``/``baseline`` leaves
    ``F0`` at 0, which callers must treat as "not calibrated".
    """
    n = len(footprints)
    gain = np.ones(n, dtype=np.float32)
    f0 = np.zeros(n, dtype=np.float32)

    var = None if var_img is None else np.asarray(var_img, dtype=np.float64).ravel()
    mean = None if mean_img is None else np.asarray(mean_img, dtype=np.float64).ravel()
    base = None if baseline is None else np.asarray(baseline, dtype=np.float64).ravel()

    for k, (pix, lam) in enumerate(footprints):
        if pix.size == 0:
            continue
        idx = pix.astype(np.int64)
        scale = lam.astype(np.float64)
        if var is not None:
            scale = scale * var[idx]
        gain[k] = float(scale.mean())

        level = np.zeros(idx.size, dtype=np.float64)
        if base is not None:
            level += base[idx] * (var[idx] if var is not None else 1.0)
        if mean is not None:
            level += mean[idx]
        f0[k] = float(level.mean())

    return gain, f0


def calibrated_traces(
    c: np.ndarray,
    gain: np.ndarray,
    f0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(F, dff)`` from demixed ``c`` (T, K) and a :func:`roi_calibration`.

    ``F`` is (K, T) in movie units; ``dff`` is (K, T) in percent, zeroed for
    any ROI whose F0 is non-positive (uncalibrated, or a footprint sitting on
    dead pixels) so a bogus baseline never manufactures a huge transient.
    """
    amp = np.ascontiguousarray(c.T, dtype=np.float32) * gain[:, None]
    F = amp + f0[:, None]
    dff = np.zeros_like(amp)
    ok = f0 > 0
    if ok.any():
        dff[ok] = amp[ok] / f0[ok, None] * 100.0
    return F.astype(np.float32), dff.astype(np.float32)


def write_plane_outputs(
    plane_dir: str | Path,
    *,
    indices: np.ndarray,
    values: np.ndarray,
    c: np.ndarray,
    shape: tuple[int, int],
    baseline: np.ndarray | None = None,
    var_img: np.ndarray | None = None,
    mean_img: np.ndarray | None = None,
) -> dict:
    """Write stat/iscell/F/Fneu/spks/norm_traces for one plane.

    ``c`` is (T, K) demixed traces; ``indices``/``values`` the COO spatial
    footprints; ``baseline`` masknmf's static ``b``; ``var_img``/``mean_img``
    the PMD standardisation images.

    ``F.npy`` is calibrated back to movie units and ``norm_traces.npy`` is
    percent dF/F against masknmf's own F0 — *not* a z-score. A z-score would
    subtract ``mean(c)``, but masknmf already factored the baseline out into
    ``b``, so ``mean(c)`` is a function of how often the source was active,
    not of resting fluorescence; and ``std(c)`` of a nonnegative, mostly-zero
    trace measures its transients, not its noise (HALS pushed the noise into
    the residual movie). Both would make cross-ROI and cross-channel
    amplitudes depend on event rate. See :func:`roi_calibration`.

    Returns summary counts for logging/ops.
    """
    plane_dir = Path(plane_dir)
    n_rois = int(c.shape[1]) if c.ndim == 2 else 0
    footprints = split_sparse_footprints(indices, values, n_rois)

    gain, f0 = roi_calibration(
        footprints, var_img=var_img, mean_img=mean_img, baseline=baseline
    )
    F, dff = calibrated_traces(c, gain, f0)
    n_calibrated = int((f0 > 0).sum())

    stat = np.array(
        [
            roi_stat(pix, lam, shape, F[k])
            for k, (pix, lam) in enumerate(footprints)
        ],
        dtype=object,
    )
    # masknmf has no accept/reject classifier: everything demixed is accepted
    iscell = np.ones((n_rois, 2), dtype=np.float32)

    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "iscell.npy", iscell)
    np.save(plane_dir / "F.npy", F)
    np.save(plane_dir / "Fneu.npy", np.zeros_like(F))
    np.save(plane_dir / "spks.npy", np.zeros_like(F))
    np.save(plane_dir / "norm_traces.npy", dff)
    return {"n_rois": n_rois, "n_calibrated": n_calibrated}


def merge_ops(plane_dir: str | Path, updates: dict) -> dict:
    """Merge ``updates`` into the plane dir's ops.npy (create if missing)."""
    from mbo_utilities.metadata.base import normalize_ops_arrays

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
    np.save(ops_path, normalize_ops_arrays(ops))
    return ops
