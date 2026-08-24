"""QC figures for masknmf plane and volume outputs.

Where LBM-Suite2p-Python is installed, its numbered per-plane figure suite and
volume aggregates run unchanged against the suite2p-shaped sidecars the runner
writes — same filenames, same metric formulas, directly comparable with
suite2p runs of the same data. masknmf-native panels fill the gaps lsp leaves
open: 06_registration.png (an empty slot in lsp's numbering) and PMD
compression diagnostics. Figure failures log and continue; QC never aborts a
run, matching lsp's behavior.
"""

import importlib.util
from pathlib import Path

import numpy as np

from mbo_utilities import log

_FIG_BG = "black"
_FIG_FG = "white"


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
    return (
        importlib.util.find_spec("lbm_suite2p_python") is not None
        and importlib.util.find_spec("suite2p") is not None
    )


def plot_registration_summary(plane_dir: Path, shifts: np.ndarray, save_name="06_registration.png"):
    """Shift traces + displacement histogram for rigid or piecewise shifts."""
    plt = _agg_plt()
    shifts = np.asarray(shifts)
    if shifts.ndim == 4:  # piecewise: (T, K1, K2, 2)
        per_frame = shifts.reshape(shifts.shape[0], -1, 2)
        dy, dx = per_frame[..., 0], per_frame[..., 1]
        dy_mean, dx_mean = dy.mean(axis=1), dx.mean(axis=1)
        dy_max = np.abs(dy).max(axis=1)
        dx_max = np.abs(dx).max(axis=1)
    else:  # rigid: (T, 2)
        dy_mean, dx_mean = shifts[:, 0], shifts[:, 1]
        dy_max = dx_max = None

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
    ax.set_title("Motion correction shifts", color=_FIG_FG)

    ax = axes[1]
    mag = np.sqrt(dy_mean**2 + dx_mean**2)
    ax.hist(mag, bins=50, color="#9b59b6")
    ax.set_xlabel("displacement (px)")
    ax.set_ylabel("frames")

    for ax in axes:
        ax.set_facecolor(_FIG_BG)
        ax.tick_params(colors=_FIG_FG)
        for spine in ax.spines.values():
            spine.set_color(_FIG_FG)
        ax.xaxis.label.set_color(_FIG_FG)
        ax.yaxis.label.set_color(_FIG_FG)
    fig.tight_layout()
    fig.savefig(Path(plane_dir) / save_name, dpi=150, facecolor=_FIG_BG)
    plt.close(fig)


def plot_pmd_diagnostics(plane_dir: Path, pmd, save_name="16_pmd_diagnostics.png"):
    """Mean / variance images and the per-block PMD rank heatmap."""
    plt = _agg_plt()

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    panels = []
    mean_img = getattr(pmd, "mean_img", None)
    var_img = getattr(pmd, "var_img", None)
    if mean_img is not None:
        panels.append(("mean image", _np(mean_img), "gray"))
    if var_img is not None:
        panels.append(("noise normalizer", _np(var_img), "magma"))
    try:
        panels.append(("PMD rank per block", _np(pmd.calculate_rank_heatmap()), "viridis"))
    except Exception:
        pass
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), facecolor=_FIG_BG)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img, cmap) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, color=_FIG_FG)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.045)
    rank = getattr(pmd, "pmd_rank", None)
    if rank is not None:
        fig.suptitle(f"PMD rank {int(rank)}", color=_FIG_FG)
    fig.tight_layout()
    fig.savefig(Path(plane_dir) / save_name, dpi=150, facecolor=_FIG_BG)
    plt.close(fig)


def plot_plane_figures(
    plane_dir: str | Path,
    shifts: np.ndarray | None = None,
    pmd=None,
    results=None,
    fs: float | None = None,
    logger=None,
) -> None:
    """All per-plane QC: masknmf-native panels + the lsp figure suite."""
    logger = logger or log.get()
    plane_dir = Path(plane_dir)

    if pmd is not None:
        _try(logger, "pmd diagnostics", plot_pmd_diagnostics, plane_dir, pmd)

    if results is not None and _has_lsp():

        def _roi_stats():
            from lbm_suite2p_python.postprocessing import compute_roi_stats

            compute_roi_stats(plane_dir, fs=fs)

        def _zplane_figs():
            from lbm_suite2p_python.zplane import plot_zplane_figures

            plot_zplane_figures(plane_dir, norm_method="zscore", correct_neuropil=False)

        _try(logger, "roi stats", _roi_stats)
        _try(logger, "zplane figures", _zplane_figs)
    elif results is not None:
        logger.info(
            "masknmf qc: lbm_suite2p_python not installed - "
            "skipping the shared figure suite"
        )

    # after the lsp suite: plot_zplane_figures unlinks 06_registration.png
    # (a slot it never regenerates), so the masknmf panel must come last
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
