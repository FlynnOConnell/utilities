"""QC figures for masknmf plane and volume outputs.

Figure names describe what the array actually is, not the suite2p slot it
happens to occupy. A masknmf plane is not a suite2p plane: there is no
neuropil trace, no deconvolution, no accept/reject classifier, and ``F`` is a
demixed NMF component rather than a weighted pixel average. Names copied from
suite2p ("01_correlation", "05_quality_diagnostics") hid all of that.

LBM-Suite2p-Python still draws the trace, projection and ROI-metric panels —
its plotting is good and worth reusing — but its outputs are renamed to the
masknmf vocabulary afterwards (see ``_LSP_RENAMES``). The suite2p-shaped
sidecars on disk keep their suite2p names, so lsp, ``Suite2pArray`` and
``mbo info`` all keep working unchanged.

Figures unique to masknmf are drawn natively here: the two correlation images
(signal and residual), footprint coverage, the per-ROI signal / background /
residual decomposition, calibrated dF/F, registration shifts and the PMD
basis.

Figure failures log and continue; QC never aborts a run, matching lsp.
"""

import importlib.util
from pathlib import Path

import numpy as np

from mbo_utilities import log

_FIG_BG = "black"
_FIG_FG = "white"

# lsp writes suite2p-slot names; these are what they actually contain for a
# masknmf plane. Applied after the lsp suite runs.
# Every summary image is immediately followed by its footprint overlay: the
# "<name>.png" / "<name>_footprints.png" pair sorts adjacently because '.'
# precedes '_' in ASCII.
_LSP_RENAMES = {
    "01_correlation_segmentation.png": "01a_correlation_signal_max_footprints.png",
    "03_mean.png": "02_registered_mean.png",
    "03_mean_segmentation.png": "02_registered_mean_footprints.png",
    "02_max_projection.png": "03_registered_max_projection.png",
    "02_max_projection_segmentation.png": "03_registered_max_projection_footprints.png",
    "05_quality_diagnostics.png": "07_roi_quality_metrics.png",
    "07a_traces_raw_20.png": "08a_demixed_traces_20.png",
    "07b_traces_raw_50.png": "08b_demixed_traces_50.png",
    "07c_traces_raw_100.png": "08c_demixed_traces_100.png",
    "08a_traces_norm_20.png": "09a_demixed_traces_zscore_20.png",
    "08b_traces_norm_50.png": "09b_demixed_traces_zscore_50.png",
    "08c_traces_norm_100.png": "09c_demixed_traces_zscore_100.png",
    "13_regional_zoom.png": "12_footprints_regional_zoom.png",
}

# lsp figures with no meaning for a masknmf plane. The rejected-ROI panels are
# always empty (every demixed component is accepted). The shot-noise panels
# divide by a percentile baseline a nonnegative NMF component does not have,
# and frame-to-frame differences of an already-denoised trace do not measure
# shot noise; residual RMS in the signal-decomposition figure is the honest
# noise readout for demixed data.
_LSP_DROP = (
    "01_correlation.png",  # superseded by the native 01a/01b pair
    "04b_rejected_segmentation.png",
    "09_traces_rejected.png",
    "10_shot_noise_accepted.png",
    "11_shot_noise_rejected.png",
)

_NATIVE_FIGURES = (
    "01a_correlation_signal_max.png",
    "01b_correlation_residual.png",
    "01b_correlation_residual_footprints.png",
    "04_registration_shifts.png",
    "05_pmd_basis_diagnostics.png",
    "06_footprint_coverage.png",
    "10_demixed_traces_dff.png",
    "11_roi_signal_decomposition.png",
)

# names this module used in earlier versions; cleared so a plane dir processed
# by an older version does not keep duplicates under the old slot names
_LEGACY_FIGURES = (
    "06_registration.png",
    "16_pmd_diagnostics.png",
    "02_correlation_signal_max_footprints.png",
    "03_registered_mean.png",
    "04_registered_mean_footprints.png",
    "05_registered_max_projection.png",
    "06_registered_max_projection_footprints.png",
    "07_registration_shifts.png",
    "08_pmd_basis_diagnostics.png",
    "09_footprint_coverage.png",
    "10_roi_quality_metrics.png",
    "11a_demixed_traces_20.png",
    "11b_demixed_traces_50.png",
    "11c_demixed_traces_100.png",
    "12a_demixed_traces_zscore_20.png",
    "12b_demixed_traces_zscore_50.png",
    "12c_demixed_traces_zscore_100.png",
    "13_demixed_traces_dff.png",
    "14_roi_signal_decomposition.png",
    "15_footprints_regional_zoom.png",
)


def _agg_plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _try(logger, label, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        (logger or log.get()).warning(f"masknmf qc: {label} failed: {e}")
        return None


def _has_lsp() -> bool:
    return importlib.util.find_spec("lbm_suite2p_python") is not None


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _dark(ax):
    ax.set_facecolor(_FIG_BG)
    ax.tick_params(colors=_FIG_FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_FIG_FG)
    ax.xaxis.label.set_color(_FIG_FG)
    ax.yaxis.label.set_color(_FIG_FG)
    ax.title.set_color(_FIG_FG)


def _save(fig, path):
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=150, facecolor=_FIG_BG, bbox_inches="tight")
    plt.close(fig)


def _cbar(fig, im, ax, label=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.045)
    cb.ax.tick_params(colors=_FIG_FG, labelsize=8)
    if label:
        cb.set_label(label, color=_FIG_FG, fontsize=8)
    return cb


def _sparse_footprints(results):
    """(pixel_idx, roi_idx, lam) arrays from the demixed spatial matrix ``a``."""
    a = results.a
    coo = a.coalesce() if hasattr(a, "coalesce") else a
    idx = _np(coo.indices())
    val = _np(coo.values())
    return idx[0].astype(np.int64), idx[1].astype(np.int64), val.astype(np.float64)


# --------------------------------------------------------------------------
# correlation images (demixing stage)
# --------------------------------------------------------------------------


def _summary_image(plane_dir: Path, img, title: str, save_name: str):
    """Render one full-FOV summary image, matching the lsp projection style."""
    plt = _agg_plt()
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=_FIG_BG)
    ax.set_facecolor(_FIG_BG)
    ax.imshow(_np(img), cmap="gray")
    ax.set_title(title, color=_FIG_FG, fontweight="bold", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(plane_dir) / save_name, dpi=300, facecolor=_FIG_BG)
    plt.close(fig)


def plot_signal_correlation_image(
    plane_dir: Path, vcorr, save_name="01a_correlation_signal_max.png"
):
    """Per-pixel best correlation with any demixed signal.

    masknmf builds one Pearson correlation image per demixed trace — every
    pixel's time course in the PMD-denoised movie against that trace — and
    this is their per-pixel maximum over all K signals.

    It is *not* suite2p's Vcorr. Suite2p correlates each pixel with its
    spatial neighbours in the registered movie *before* extraction, so bright
    pixels there flag cells you have not found yet. This image is derived from
    the extracted traces, so it can only be bright where a signal already
    exists, and taking a max over K maps inflates background pixels (best of K
    draws). Read it as "footprint support", not "cells present"; 01b is the
    map that shows what was missed.
    """
    _summary_image(
        plane_dir,
        vcorr,
        "Best correlation of each pixel with any demixed signal",
        save_name,
    )


def plot_residual_correlation_image(
    plane_dir: Path, resid_img, save_name="01b_correlation_residual.png"
):
    """Correlation structure left in the residual after demixing.

    Computed on the residual movie (PMD minus signals, background and
    baseline), so bright regions are coherent activity the K extracted
    components did *not* explain — missed cells, split components, or
    unmodelled background. This is the map that answers "did demixing miss
    anything", the role suite2p's Vcorr plays in its own pipeline.
    """
    _summary_image(
        plane_dir,
        resid_img,
        "Correlated activity left unexplained after demixing",
        save_name,
    )


def plot_residual_correlation_footprints(
    plane_dir: Path,
    resid_img,
    save_name="01b_correlation_residual_footprints.png",
):
    """Residual correlation image with the demixed footprints drawn on top.

    Signal in the residual that sits *outside* every footprint is a missed
    source; signal *inside* a footprint is an under-fit component.
    """
    from lbm_suite2p_python.zplane import plot_masks

    stat = np.load(Path(plane_dir) / "stat.npy", allow_pickle=True)
    iscell = np.load(Path(plane_dir) / "iscell.npy", allow_pickle=True)
    mask = iscell[:, 0].astype(bool)
    plot_masks(
        img=_np(resid_img),
        stat=stat,
        mask_idx=mask,
        savepath=Path(plane_dir) / save_name,
        title="Unexplained residual activity vs demixed footprints",
    )


# --------------------------------------------------------------------------
# footprints
# --------------------------------------------------------------------------


def plot_footprint_coverage(
    plane_dir: Path, results, shape, save_name="06_footprint_coverage.png"
):
    """Where the demixer put spatial mass, and how much the footprints overlap."""
    plt = _agg_plt()
    pix, roi, lam = _sparse_footprints(results)
    h, w = int(shape[0]), int(shape[1])
    n_rois = int(roi.max()) + 1 if roi.size else 0

    weight = np.zeros(h * w)
    np.add.at(weight, pix, lam)
    count = np.zeros(h * w, dtype=np.int32)
    np.add.at(count, pix, (lam != 0).astype(np.int32))
    npix = np.bincount(roi, minlength=n_rois)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor=_FIG_BG)

    im = axes[0].imshow(weight.reshape(h, w), cmap="viridis")
    axes[0].set_title("Summed footprint weight (lam)", color=_FIG_FG, fontsize=10)
    axes[0].axis("off")
    _cbar(fig, im, axes[0])

    im = axes[1].imshow(count.reshape(h, w), cmap="magma", vmin=0)
    axes[1].set_title(
        f"Signals per pixel — max {int(count.max())}", color=_FIG_FG, fontsize=10
    )
    axes[1].axis("off")
    _cbar(fig, im, axes[1])

    ax = axes[2]
    ax.hist(npix, bins=40, color="#2ecc71")
    ax.set_xlabel("pixels per footprint")
    ax.set_ylabel("signals")
    ax.set_title("Footprint size", fontsize=10)
    _dark(ax)

    overlap = float((count > 1).sum() / max((count > 0).sum(), 1))
    fig.suptitle(
        f"Footprint coverage — {n_rois} signals, {(count > 0).mean():.1%} of FOV covered, "
        f"median {int(np.median(npix))} px each, {overlap:.1%} of covered pixels shared",
        color=_FIG_FG,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, Path(plane_dir) / save_name)


# --------------------------------------------------------------------------
# per-ROI signal decomposition
# --------------------------------------------------------------------------


def _roi_averages(results):
    """(pmd, fluctuating background, residual) ROI-average traces, each (K, T)."""
    if getattr(results, "pmd_roi_averages", None) is None:
        results._set_roi_averages()
    return (
        _np(results.pmd_roi_averages),
        _np(results.fluctuating_background_roi_averages),
        _np(results.residual_roi_averages),
    )


def plot_roi_signal_decomposition(
    plane_dir: Path,
    results,
    fs: float | None = None,
    save_name="11_roi_signal_decomposition.png",
):
    """Split each ROI's movie signal into demixed / background / residual.

    masknmf has no neuropil trace, so ``Fneu.npy`` is written as zeros. The
    real analogue is the ring-model fluctuating background averaged over the
    ROI support, and masknmf enforces the exact identity

        pmd_roi_average = demixed + static bkgd + fluctuating bkgd + residual

    The variance each term explains is the honest contamination and
    goodness-of-fit readout for a demixed plane, and residual RMS is the
    honest noise readout — shot noise is undefined on an already-denoised
    nonnegative component.
    """
    plt = _agg_plt()
    pmd_avg, bkgd_avg, resid_avg = _roi_averages(results)
    c = _np(results.c)  # (T, K)
    pix, roi, lam = _sparse_footprints(results)
    n_rois = c.shape[1]

    # ac_roi_average: the same binarised-support average applied to a @ c.T
    counts = np.bincount(roi, minlength=n_rois).astype(np.float64)
    lam_sum = np.zeros(n_rois)
    np.add.at(lam_sum, roi, lam)
    ac_avg = (lam_sum / np.maximum(counts, 1))[:, None] * c.T

    v_pmd = np.maximum(np.var(pmd_avg, axis=1), 1e-12)
    frac_sig = np.var(ac_avg, axis=1) / v_pmd
    frac_bkg = np.var(bkgd_avg, axis=1) / v_pmd
    frac_res = np.var(resid_avg, axis=1) / v_pmd
    resid_rms = np.sqrt(np.mean(resid_avg**2, axis=1))

    n_t = pmd_avg.shape[1]
    t = np.arange(n_t) / fs if fs else np.arange(n_t)
    xlabel = "time (s)" if fs else "frame"
    seg = slice(0, min(2000, n_t))

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), facecolor=_FIG_BG)

    order = np.argsort(-frac_sig)
    picks = [order[0], order[len(order) // 2], order[-1]]
    labels = ["highest signal fraction", "median", "lowest"]
    for ax, k, lab in zip(axes[0], picks, labels):
        ax.plot(t[seg], pmd_avg[k][seg], color="#dddddd", lw=0.6, label="PMD ROI avg")
        ax.plot(t[seg], ac_avg[k][seg], color="#2ecc71", lw=0.9, label="demixed")
        ax.plot(t[seg], bkgd_avg[k][seg], color="#3498db", lw=0.6, label="background")
        ax.plot(
            t[seg], resid_avg[k][seg], color="#e74c3c", lw=0.4, alpha=0.7, label="residual"
        )
        ax.set_title(
            f"signal {k} — {lab}\nvar: sig {frac_sig[k]:.2f} | bkgd {frac_bkg[k]:.2f} | "
            f"resid {frac_res[k]:.2f}",
            fontsize=9,
        )
        ax.set_xlabel(xlabel)
        _dark(ax)
    axes[0][0].legend(
        loc="upper right",
        fontsize=7,
        facecolor=_FIG_BG,
        edgecolor="gray",
        labelcolor=_FIG_FG,
    )

    ax = axes[1][0]
    bins = np.linspace(0, min(2.0, float(np.percentile(frac_sig, 99.5)) + 0.1), 45)
    ax.hist(frac_sig, bins=bins, color="#2ecc71", alpha=0.8, label="signal")
    ax.hist(frac_bkg, bins=bins, color="#3498db", alpha=0.6, label="background")
    ax.hist(frac_res, bins=bins, color="#e74c3c", alpha=0.6, label="residual")
    ax.set_xlabel("variance fraction of PMD ROI average")
    ax.set_ylabel("signals")
    ax.set_title("Variance decomposition", fontsize=10)
    ax.legend(fontsize=7, facecolor=_FIG_BG, edgecolor="gray", labelcolor=_FIG_FG)
    _dark(ax)

    ax = axes[1][1]
    sc = ax.scatter(frac_bkg, frac_sig, s=7, c=frac_res, cmap="inferno")
    ax.set_xlabel("background variance fraction")
    ax.set_ylabel("signal variance fraction")
    ax.set_title("Background contamination", fontsize=10)
    _cbar(fig, sc, ax, "residual fraction")
    _dark(ax)

    ax = axes[1][2]
    ax.hist(resid_rms, bins=40, color="#9b59b6")
    ax.set_xlabel("residual RMS (PMD units)")
    ax.set_ylabel("signals")
    ax.set_title("Per-signal residual noise", fontsize=10)
    _dark(ax)

    fig.suptitle(
        f"ROI signal decomposition — {n_rois} signals   median variance fraction: "
        f"signal {np.median(frac_sig):.2f} | background {np.median(frac_bkg):.2f} | "
        f"residual {np.median(frac_res):.2f}",
        color=_FIG_FG,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, Path(plane_dir) / save_name)


# --------------------------------------------------------------------------
# calibrated dF/F
# --------------------------------------------------------------------------


def plot_calibrated_dff(
    plane_dir: Path,
    results,
    pmd,
    fs: float | None = None,
    n_traces: int = 20,
    save_name="10_demixed_traces_dff.png",
):
    """dF/F with a real F0, recovered from the PMD standardisation.

    The demixer runs on ``(movie - mean_img) / var_img``, so ``c`` is in
    standardised units and ``F.npy`` carries no F0 — a percentile baseline of
    a nonnegative component is ~0, which is why a generic percentile-dF/F path
    ends up dividing by its epsilon guard. Undoing the standardisation per ROI
    with the lam-weighted PMD images restores movie units::

        gain_k = sum(lam * var_img) / sum(lam)      # amplitude scale
        F0_k   = sum(lam * mean_img) / sum(lam)     # baseline
        dF/F   = c_k * gain_k / F0_k
    """
    plt = _agg_plt()
    pix, roi, lam = _sparse_footprints(results)
    mean_img = _np(pmd.mean_img).ravel()
    var_img = _np(pmd.var_img).ravel()
    c = _np(results.c)  # (T, K)
    n_rois = c.shape[1]

    lam_sum = np.zeros(n_rois)
    gain = np.zeros(n_rois)
    f0 = np.zeros(n_rois)
    np.add.at(lam_sum, roi, lam)
    np.add.at(gain, roi, lam * var_img[pix])
    np.add.at(f0, roi, lam * mean_img[pix])
    ok = lam_sum > 0
    gain[ok] /= lam_sum[ok]
    f0[ok] /= lam_sum[ok]

    dff = (c.T * gain[:, None]) / np.maximum(f0, 1e-6)[:, None] * 100.0
    peak = dff.max(axis=1)
    order = np.argsort(-peak)[:n_traces]

    n_t = dff.shape[1]
    t = np.arange(n_t) / fs if fs else np.arange(n_t)
    xlabel = "time (s)" if fs else "frame"

    fig, axes = plt.subplots(
        1, 2, figsize=(17, 8), facecolor=_FIG_BG, gridspec_kw={"width_ratios": [3, 1]}
    )

    ax = axes[0]
    step = float(np.percentile(peak[order], 75)) * 0.7 or 1.0
    for i, k in enumerate(order):
        ax.plot(
            t,
            dff[k] + i * step,
            lw=0.5,
            color=plt.cm.viridis(i / max(len(order) - 1, 1)),
        )
    ax.plot([t[0], t[0]], [-step * 0.5, -step * 0.5 + 100], color=_FIG_FG, lw=2.5)
    ax.text(
        t[max(1, n_t // 200)],
        -step * 0.5 + 50,
        "100% dF/F",
        color=_FIG_FG,
        fontsize=8,
        va="center",
    )
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_title(
        f"Top {len(order)} demixed signals by peak dF/F", fontsize=11
    )
    _dark(ax)

    ax = axes[1]
    ax.hist(peak, bins=40, color="#e67e22")
    ax.set_xlabel("peak dF/F (%)")
    ax.set_ylabel("signals")
    ax.set_title("Peak amplitude, all signals", fontsize=10)
    _dark(ax)

    fig.suptitle(
        f"Calibrated dF/F — median peak {np.median(peak):.0f}%, "
        f"p90 {np.percentile(peak, 90):.0f}%   "
        "(F0 and gain from lam-weighted PMD mean / noise images)",
        color=_FIG_FG,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, Path(plane_dir) / save_name)


# --------------------------------------------------------------------------
# registration and PMD
# --------------------------------------------------------------------------


def plot_registration_summary(
    plane_dir: Path, shifts: np.ndarray, save_name="04_registration_shifts.png"
):
    """Shift traces + displacement histogram for rigid or piecewise shifts."""
    plt = _agg_plt()
    shifts = np.asarray(shifts)
    if shifts.ndim == 4:  # piecewise: (T, K1, K2, 2)
        per_frame = shifts.reshape(shifts.shape[0], -1, 2)
        dy, dx = per_frame[..., 0], per_frame[..., 1]
        dy_mean, dx_mean = dy.mean(axis=1), dx.mean(axis=1)
        dy_max = np.abs(dy).max(axis=1)
        dx_max = np.abs(dx).max(axis=1)
        kind = "piecewise-rigid"
    else:  # rigid: (T, 2)
        dy_mean, dx_mean = shifts[:, 0], shifts[:, 1]
        dy_max = dx_max = None
        kind = "rigid"

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), facecolor=_FIG_BG)
    ax = axes[0]
    ax.plot(dy_mean, color="#2ecc71", lw=0.7, label="dy")
    ax.plot(dx_mean, color="#3498db", lw=0.7, label="dx")
    if dy_max is not None:
        ax.plot(dy_max, color="#2ecc71", lw=0.4, alpha=0.4, label="|dy| max block")
        ax.plot(dx_max, color="#3498db", lw=0.4, alpha=0.4, label="|dx| max block")
    ax.set_xlabel("frame")
    ax.set_ylabel("shift (px)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Motion correction shifts ({kind})", fontsize=10)

    ax = axes[1]
    mag = np.sqrt(dy_mean**2 + dx_mean**2)
    ax.hist(mag, bins=50, color="#9b59b6")
    ax.set_xlabel("displacement (px)")
    ax.set_ylabel("frames")
    ax.set_title(
        f"median {np.median(mag):.2f} px, 99th pct {np.percentile(mag, 99):.2f} px",
        fontsize=9,
    )

    for ax in axes:
        _dark(ax)
    fig.tight_layout()
    _save(fig, Path(plane_dir) / save_name)


def plot_pmd_diagnostics(plane_dir: Path, pmd, save_name="05_pmd_basis_diagnostics.png"):
    """Mean / noise-normalizer images and the per-pixel PMD rank heatmap."""
    plt = _agg_plt()

    panels = []
    mean_img = getattr(pmd, "mean_img", None)
    var_img = getattr(pmd, "var_img", None)
    if mean_img is not None:
        panels.append(("mean image (movie units)", _np(mean_img), "gray"))
    if var_img is not None:
        panels.append(("noise normalizer (per-pixel sigma)", _np(var_img), "magma"))
    try:
        panels.append(
            (
                "components per pixel (rank heatmap)",
                _np(pmd.calculate_rank_heatmap()),
                "viridis",
            )
        )
    except Exception:
        pass
    if not panels:
        return

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.8 * len(panels), 5.2), facecolor=_FIG_BG
    )
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img, cmap) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, color=_FIG_FG, fontsize=10)
        ax.axis("off")
        _cbar(fig, im, ax)
    rank = getattr(pmd, "pmd_rank", None)
    if rank is not None:
        fig.suptitle(
            f"PMD basis — total rank {int(rank)}",
            color=_FIG_FG,
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, Path(plane_dir) / save_name)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _clear_previous(plane_dir: Path) -> None:
    """Remove figures this module owns.

    lsp only unlinks its own slot names, so renamed outputs from an earlier
    run would otherwise survive as stale duplicates.
    """
    for name in (
        list(_LSP_RENAMES.values())
        + list(_LSP_DROP)
        + list(_NATIVE_FIGURES)
        + list(_LEGACY_FIGURES)
    ):
        p = plane_dir / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


def _rename_lsp_outputs(plane_dir: Path, logger) -> None:
    for old, new in _LSP_RENAMES.items():
        src = plane_dir / old
        if not src.exists():
            continue
        dst = plane_dir / new
        try:
            if dst.exists():
                dst.unlink()
            src.rename(dst)
        except OSError as e:
            logger.warning(f"masknmf qc: could not rename {old} -> {new}: {e}")
    for name in _LSP_DROP:
        p = plane_dir / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


def plot_plane_figures(
    plane_dir: str | Path,
    shifts: np.ndarray | None = None,
    pmd=None,
    results=None,
    fs: float | None = None,
    logger=None,
) -> None:
    """All per-plane QC: the renamed lsp suite plus masknmf-native panels."""
    logger = logger or log.get()
    plane_dir = Path(plane_dir)
    _clear_previous(plane_dir)

    if results is not None and _has_lsp():

        def _roi_stats():
            from lbm_suite2p_python.postprocessing import compute_roi_stats

            compute_roi_stats(plane_dir, fs=fs)

        def _zplane_figs():
            from lbm_suite2p_python.zplane import plot_zplane_figures

            plot_zplane_figures(plane_dir, norm_method="zscore", correct_neuropil=False)

        _try(logger, "roi stats", _roi_stats)
        _try(logger, "zplane figures", _zplane_figs)
        _rename_lsp_outputs(plane_dir, logger)
    elif results is not None:
        logger.info(
            "masknmf qc: lbm_suite2p_python not installed - "
            "skipping the shared figure suite"
        )

    if results is not None:
        vcorr = None
        ops_path = plane_dir / "ops.npy"
        if ops_path.exists():
            try:
                vcorr = np.load(ops_path, allow_pickle=True).item().get("Vcorr")
            except Exception:
                vcorr = None
        if vcorr is not None:
            _try(
                logger,
                "signal correlation image",
                plot_signal_correlation_image,
                plane_dir,
                vcorr,
            )
        resid = getattr(results, "global_residual_correlation_image", None)
        if resid is not None:
            _try(
                logger,
                "residual correlation image",
                plot_residual_correlation_image,
                plane_dir,
                resid,
            )
            if _has_lsp():
                _try(
                    logger,
                    "residual correlation footprints",
                    plot_residual_correlation_footprints,
                    plane_dir,
                    resid,
                )
        _try(
            logger,
            "footprint coverage",
            plot_footprint_coverage,
            plane_dir,
            results,
            results.fov_shape,
        )
        _try(
            logger,
            "roi signal decomposition",
            plot_roi_signal_decomposition,
            plane_dir,
            results,
            fs,
        )
        if pmd is not None:
            _try(logger, "calibrated dff", plot_calibrated_dff, plane_dir, results, pmd, fs)

    if pmd is not None:
        _try(logger, "pmd diagnostics", plot_pmd_diagnostics, plane_dir, pmd)

    if shifts is not None:
        _try(logger, "registration summary", plot_registration_summary, plane_dir, shifts)


def plot_volume_figures(save_path: Path, ops_files: list[Path], logger=None) -> None:
    """Volume-level QC via the lsp aggregate suite (zstats + summary PNGs)."""
    logger = logger or log.get()
    if not _has_lsp():
        logger.info("masknmf qc: lbm_suite2p_python not installed - no volume figures")
        return
    save_path = Path(save_path)
    ops_files = [str(p) for p in ops_files]

    def _stats():
        from lbm_suite2p_python.volume import get_volume_stats

        get_volume_stats(ops_files, overwrite=True)

    def _diagnostics():
        from lbm_suite2p_python.volume import plot_volume_diagnostics

        plot_volume_diagnostics(
            ops_files, save_path=str(save_path / "volume_quality_diagnostics.png")
        )

    def _ortho():
        from lbm_suite2p_python.volume import plot_orthoslices

        plot_orthoslices(ops_files, save_path=str(save_path / "orthoslices.png"))

    def _roi_map():
        from lbm_suite2p_python.volume import plot_3d_roi_map

        plot_3d_roi_map(
            ops_files, save_path=str(save_path / "roi_map_3d.png"), color_by="snr"
        )
        plot_3d_roi_map(
            ops_files,
            save_path=str(save_path / "roi_map_3d_plane.png"),
            color_by="plane",
        )

    def _overlay():
        from lbm_suite2p_python.zplane import plot_volume_accepted_rejected_overlay

        plot_volume_accepted_rejected_overlay(
            ops_files, savepath=str(save_path / "volume_segmentation_overlay.png")
        )

    def _traces():
        from lbm_suite2p_python.volume import plot_volume_trace_figures

        plot_volume_trace_figures(
            ops_files, str(save_path), norm_method="zscore", correct_neuropil=False
        )

    _try(logger, "volume stats", _stats)
    _try(logger, "volume diagnostics", _diagnostics)
    _try(logger, "orthoslices", _ortho)
    _try(logger, "3d roi map", _roi_map)
    _try(logger, "segmentation overlay", _overlay)
    _try(logger, "volume trace figures", _traces)
