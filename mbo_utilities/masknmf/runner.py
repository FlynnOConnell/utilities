"""masknmf pipeline runner: registration -> PMD compression -> demixing.

Per-plane flow mirrors lbm_suite2p_python.run_plane: write (or reuse) a
``data_raw.bin`` suite2p binary via imwrite, run the masknmf stages with
Skip/Run/Force gating on each stage's native HDF5 export
(``motion_correction.hdf5`` / ``compression.hdf5`` / ``demixing_results.hdf5``),
then export suite2p-shaped npy sidecars and QC figures. Re-running over an
existing output dir with all stages cached only redoes exports + figures.

masknmf imports are function-local: the package is optional, heavy to import,
and its stage math runs in the spawned worker subprocess.
"""

import os
import time
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.masknmf.params import (
    DEMIX_FILE,
    MOCO_FILE,
    PMD_FILE,
    MasknmfSettings,
    stage_action,
)
from mbo_utilities.masknmf import outputs as _outputs


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_plane_dirname(
    plane: int, frame_indices: list[int] | None = None
) -> str:
    """lsp-compatible plane dir name: zplaneNN[_tpAAAAA-BBBBB] (1-based)."""
    name = f"zplane{plane:02d}"
    if frame_indices:
        name += f"_tp{frame_indices[0] + 1:05d}-{frame_indices[-1] + 1:05d}"
    return name


def _history_entry(step: str, duration: float | None = None, **extra) -> dict:
    try:
        from importlib.metadata import version

        ver = version("masknmf")
    except Exception:
        ver = "unknown"
    entry = {
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "masknmf_version": ver,
    }
    if duration is not None:
        entry["duration_seconds"] = round(duration, 2)
    entry.update(extra)
    return entry


def _valid_raw_bin(plane_dir: Path, nframes: int, ly: int, lx: int) -> bool:
    raw = plane_dir / "data_raw.bin"
    if not raw.exists():
        return False
    # trust the recorded dims over the array's reported shape — multi-ROI
    # stitching can write a different height than the source reports
    ops_path = plane_dir / "ops.npy"
    if ops_path.exists():
        try:
            ops = np.load(ops_path, allow_pickle=True).item()
            ly = int(ops.get("Ly") or ly)
            lx = int(ops.get("Lx") or lx)
        except Exception:
            pass
    return raw.stat().st_size == nframes * ly * lx * 2


def _write_raw_bin(
    input_data,
    plane_dir: Path,
    plane: int,
    channel: int | None,
    frame_indices: list[int] | None,
    metadata: dict | None,
    writer_kwargs: dict | None,
    logger,
) -> Path:
    from mbo_utilities import imread, imwrite

    arr = input_data if hasattr(input_data, "shape") else imread(input_data)
    kwargs = dict(writer_kwargs or {})
    # phase correction is an array attribute, not an imwrite kwarg
    fix_phase = kwargs.pop("fix_phase", None)
    use_fft = kwargs.pop("use_fft", None)
    if fix_phase is not None and hasattr(arr, "fix_phase"):
        arr.fix_phase = bool(fix_phase)
    if use_fft is not None and hasattr(arr, "use_fft"):
        arr.use_fft = bool(use_fft)
    if frame_indices is not None:
        kwargs["timepoints"] = [i + 1 for i in frame_indices]
    if channel is not None:
        kwargs["channels"] = [int(channel)]
    # single-z sources (incl. natural-rank 3D readers) take the planeless
    # fast path; a planes= selection would force 5D indexing onto them
    nz = int(arr._shape5d()[2]) if hasattr(arr, "_shape5d") else 1
    if nz > 1:
        kwargs["planes"] = int(plane)
    imwrite(
        arr,
        plane_dir,
        ext=".bin",
        output_name="data_raw.bin",
        metadata=dict(metadata or {}),
        overwrite=True,
        **kwargs,
    )
    logger.info(f"masknmf: wrote {plane_dir / 'data_raw.bin'}")
    return plane_dir / "data_raw.bin"


def _open_raw(plane_dir: Path) -> tuple[np.memmap, dict]:
    ops = _outputs.merge_ops(plane_dir, {})
    ly = ops.get("Ly") or ops.get("ly")
    lx = ops.get("Lx") or ops.get("lx")
    if not ly or not lx:
        raise ValueError(f"{plane_dir / 'ops.npy'} lacks Ly/Lx")
    ly, lx = int(ly), int(lx)
    raw = plane_dir / "data_raw.bin"
    nframes = raw.stat().st_size // (ly * lx * 2)
    mov = np.memmap(raw, dtype=np.int16, mode="r", shape=(nframes, ly, lx))
    return mov, ops


def _stage_registration(raw, cfg, runtime, plane_dir: Path, device: str, logger):
    """Returns (movie_for_compression, shifts_np, template_np, seconds)."""
    moco_path = plane_dir / MOCO_FILE
    action = stage_action(cfg.do_registration, moco_path.exists())
    if action == "skip":
        logger.info("masknmf: registration skipped")
        return raw, None, None, 0.0

    import masknmf

    cls = (
        masknmf.PiecewiseRigidRegistrationArray
        if cfg.strategy == "pwrigid"
        else masknmf.RigidRegistrationArray
    )
    if action == "reuse":
        try:
            import torch

            moco = cls.from_hdf5(str(moco_path), input_movie=raw)
            # pwrigid ctor skips the shifts-vs-frames check the rigid one has
            if int(moco.shifts.shape[0]) != int(raw.shape[0]):
                raise ValueError(
                    f"cached shifts cover {int(moco.shifts.shape[0])} frames, "
                    f"movie has {raw.shape[0]}"
                )
            # from_hdf5 rebuilds the strategy with device='auto'; honor the
            # configured device for reads
            moco.output_device = torch.device(device)
            logger.info(f"masknmf: reusing {MOCO_FILE}")
            shifts = _to_np(moco.shifts)
            template = getattr(moco.strategy, "template", None)
            return moco, shifts, _to_np(template) if template is not None else None, 0.0
        except Exception as e:
            logger.warning(f"masknmf: cached {MOCO_FILE} unusable ({e}); recomputing")

    t0 = time.time()
    corrector_cls = (
        masknmf.PiecewiseRigidMotionCorrector
        if cfg.strategy == "pwrigid"
        else masknmf.RigidMotionCorrector
    )
    strategy = corrector_cls(
        **cfg.strategy_kwargs(),
        device=device,
        batch_size=runtime.frame_batch_size,
    )
    logger.info(f"masknmf: computing {cfg.strategy} registration template")
    strategy.compute_template(raw)
    logger.info("masknmf: estimating shifts")
    moco = strategy.motion_correct(raw)
    moco.output_device = moco.strategy.device
    if moco_path.exists():
        moco_path.unlink()
    moco.export(str(moco_path))
    shifts = _to_np(moco.shifts)
    template = getattr(moco.strategy, "template", None)
    return (
        moco,
        shifts,
        _to_np(template) if template is not None else None,
        time.time() - t0,
    )


def _shift_mask(shifts, shape: tuple[int, int], border: int) -> np.ndarray:
    mask = np.ones(shape, dtype="float")
    if shifts is not None:
        try:
            from masknmf.motion_correction.moco_preprocessing import (
                construct_moco_template,
            )

            mask = construct_moco_template(shifts, shape).astype("float")
        except Exception:
            pass
    if border > 0:
        mask[:border, :] = 0
        mask[:, :border] = 0
        mask[-border:, :] = 0
        mask[:, -border:] = 0
    return mask


def _stage_compression(
    moco, cfg, runtime, plane_dir: Path, device: str, mask, fs, logger
):
    """Returns (PMDArray, seconds)."""
    pmd_path = plane_dir / PMD_FILE
    cached = pmd_path.exists()
    action = stage_action(cfg.do_compression, cached)
    if action == "skip":
        if not cached:
            raise ValueError(
                f"compression set to Skip but {pmd_path} does not exist"
            )
        action = "reuse"

    import masknmf

    if action == "reuse":
        logger.info(f"masknmf: reusing {PMD_FILE}")
        return masknmf.PMDArray.from_hdf5(str(pmd_path)), 0.0

    t0 = time.time()
    kw = cfg.strategy_kwargs()
    kw["pixel_weighting"] = mask
    strat_cls = (
        masknmf.CompressDenoiseStrategy if cfg.denoise else masknmf.CompressStrategy
    )
    strat = strat_cls(device=device, **kw)
    nframes = int(moco.shape[0])
    if cfg.detrend and fs:
        from masknmf.compression.preprocessing import MaximinSplineDetrend

        strat.detrender = MaximinSplineDetrend(
            num_frames=nframes,
            num_knots=max(4, int(nframes / fs / 25)),
            window=int(40 * fs),
            sigma=max(2.0, 0.3 * fs),
            device=device,
        )
    logger.info("masknmf: running PMD compression")
    pmd = strat.compress(moco)
    if pmd_path.exists():
        pmd_path.unlink()
    pmd.export(str(pmd_path))
    # reload so demixing always consumes the exact persisted decomposition
    return masknmf.PMDArray.from_hdf5(str(pmd_path)), time.time() - t0


def _stage_demixing(pmd, cfg, runtime, plane_dir: Path, device: str, fs, logger):
    """Returns (DemixingResults | None, seconds)."""
    demix_path = plane_dir / DEMIX_FILE
    action = stage_action(cfg.do_demixing, demix_path.exists())
    if action == "skip":
        logger.info("masknmf: demixing skipped")
        return None, 0.0

    import masknmf

    if action == "reuse":
        logger.info(f"masknmf: reusing {DEMIX_FILE}")
        return masknmf.DemixingResults.from_hdf5(str(demix_path)), 0.0

    import torch
    from masknmf.demixing import NoSignalsDetectedError

    t0 = time.time()
    detrender = None
    if fs:
        from masknmf.compression.preprocessing import MaximinSplineDetrend

        nframes = int(pmd.shape[0])
        detrender = MaximinSplineDetrend(
            num_frames=nframes,
            num_knots=max(4, int(nframes / fs / 20)),
            window=int(20 * fs),
            sigma=max(2.0, 0.3 * fs),
            device=device,
        )

    def _empty_cache():
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    logger.info(f"masknmf: spatial highpass (sigma={cfg.filter_sigma})")
    filtered = masknmf.demixing.filters.spatial_filter_pmd(
        pmd,
        batch_size=runtime.frame_batch_size,
        filter_sigma=cfg.filter_sigma,
        device=device,
    )
    _empty_cache()

    demixer = masknmf.SignalDemixer(
        filtered, device=device, frame_batch_size=runtime.frame_batch_size
    )
    filtered_results = None
    for i in range(max(1, cfg.filtered_passes)):
        try:
            demixer.initialize_signals(**cfg.init_kwargs(detrender))
        except NoSignalsDetectedError:
            break
        logger.info(f"masknmf: demixing filtered pass {i + 1}")
        demixer.demix(
            **cfg.nmf_kwargs(cfg.support_threshold_lo, ring=False, detrender=detrender)
        )
        filtered_results = demixer.results
        _empty_cache()
    if filtered_results is None:
        raise ValueError(
            "masknmf found no signals in the highpass-filtered movie; "
            "lower mad_correlation_threshold or inspect the data"
        )

    a_init = filtered_results.ac_array.export_a()
    c_init = filtered_results.ac_array.export_c()

    demixer = masknmf.SignalDemixer(
        pmd, device=device, frame_batch_size=runtime.frame_batch_size
    )
    results = None
    for i in range(max(1, cfg.unfiltered_passes)):
        try:
            if i == 0:
                demixer.initialize_signals(
                    is_custom=True,
                    spatial_footprints=a_init,
                    temporal_footprints=c_init,
                    c_nonneg=True,
                )
            else:
                demixer.initialize_signals(**cfg.init_kwargs(detrender))
        except NoSignalsDetectedError:
            break
        logger.info(f"masknmf: demixing unfiltered pass {i + 1}")
        demixer.demix(
            **cfg.nmf_kwargs(cfg.unfiltered_support_lo, ring=True, detrender=detrender)
        )
        results = demixer.results
        _empty_cache()
    if results is None:
        raise ValueError("masknmf unfiltered demixing did not complete a pass")

    if demix_path.exists():
        demix_path.unlink()
    results.export(str(demix_path))
    return results, time.time() - t0


def _write_registered_bin(moco, plane_dir: Path, logger) -> tuple[np.ndarray, np.ndarray]:
    """Stream the registered movie to data.bin; returns (meanImg, max_proj).

    Written to a temp name and renamed on completion: np.memmap
    preallocates the full file up front, so a crash mid-write would
    otherwise leave a full-size partial data.bin that later runs (and
    Suite2pArray) trust as complete.
    """
    nframes, ly, lx = (int(s) for s in moco.shape)
    tmp = plane_dir / "data.bin.partial"
    out = np.memmap(tmp, dtype=np.int16, mode="w+", shape=(nframes, ly, lx))
    acc = np.zeros((ly, lx), dtype=np.float64)
    mx = np.full((ly, lx), -np.inf, dtype=np.float64)
    step = 200
    for t0 in range(0, nframes, step):
        # masknmf arrays raise on slice stops past n_frames instead of
        # clamping like numpy
        chunk = _to_np(moco[t0: min(t0 + step, nframes)]).astype(np.float32)
        acc += chunk.sum(axis=0)
        np.maximum(mx, chunk.max(axis=0), out=mx)
        out[t0: t0 + chunk.shape[0]] = np.clip(chunk, -32768, 32767).astype(np.int16)
    out.flush()
    del out
    os.replace(tmp, plane_dir / "data.bin")
    logger.info(f"masknmf: wrote {plane_dir / 'data.bin'}")
    return (acc / max(nframes, 1)).astype(np.float32), mx.astype(np.float32)


def _movie_stats(moco) -> tuple[np.ndarray, np.ndarray]:
    """meanImg/max_proj from up to 500 evenly-sampled frames (no bin write)."""
    nframes = int(moco.shape[0])
    idx = np.unique(np.linspace(0, nframes - 1, min(nframes, 500)).astype(int))
    frames = _to_np(moco[idx.tolist()]).astype(np.float32)
    return frames.mean(axis=0), frames.max(axis=0)


def _extract_footprints(results):
    """(indices (2,nnz), values, baseline) numpy from DemixingResults."""
    ac = results.ac_array
    sparse = None
    for obj, attr in ((results, "a"), (ac, "a"), (ac, "_a")):
        t = getattr(obj, attr, None)
        if t is not None and hasattr(t, "coalesce"):
            sparse = t
            break
    if sparse is not None:
        coo = sparse.coalesce()
        indices = _to_np(coo.indices())
        values = _to_np(coo.values())
    else:
        dense = ac.export_a()  # (H, W, K)
        h, w, k = dense.shape
        flat = dense.reshape(h * w, k)
        pix, roi = np.nonzero(flat)
        indices = np.stack([pix, roi])
        values = flat[pix, roi]
    baseline = getattr(results, "b", None)
    return indices, values, _to_np(baseline) if baseline is not None else None


def run_plane(
    input_data,
    save_path,
    plane: int = 1,
    settings: MasknmfSettings | dict | None = None,
    metadata: dict | None = None,
    frame_indices: list[int] | None = None,
    channel: int | None = None,
    plane_name: str | None = None,
    replot: bool = True,
    writer_kwargs: dict | None = None,
    logger=None,
    progress_callback=None,
) -> Path:
    """Run the masknmf pipeline on one z-plane (1-based). Returns ops.npy path."""
    logger = logger or log.get()
    if not isinstance(settings, MasknmfSettings):
        settings = MasknmfSettings.from_dict(settings)
    reg, comp, demix, runtime = (
        settings.registration,
        settings.compression,
        settings.demixing,
        settings.runtime,
    )
    save_path = Path(save_path)
    plane_dir = save_path / (plane_name or generate_plane_dirname(plane, frame_indices))
    plane_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(runtime.device)
    total_t0 = time.time()

    def _progress(step: str, message: str):
        if progress_callback is not None:
            try:
                progress_callback(step=step, message=message)
            except Exception:
                pass

    # 1) raw binary (reused when dims already match, like lsp _should_write_bin)
    from mbo_utilities import imread
    from mbo_utilities.metadata import get_param

    arr = input_data if hasattr(input_data, "shape") else imread(input_data)
    src_ly, src_lx = int(arr.shape[-2]), int(arr.shape[-1])
    want_frames = (
        len(frame_indices) if frame_indices is not None else int(arr.shape[0])
    )
    force_bin = reg.do_registration == 2 or comp.do_compression == 2
    if force_bin or not _valid_raw_bin(plane_dir, want_frames, src_ly, src_lx):
        _progress("writing_binary", f"Writing raw binary for plane {plane}")
        _write_raw_bin(
            arr, plane_dir, plane, channel, frame_indices, metadata, writer_kwargs, logger
        )
    else:
        logger.info("masknmf: reusing existing data_raw.bin")

    raw, ops = _open_raw(plane_dir)
    fs = get_param(ops, "fs") or get_param(dict(metadata or {}), "fs")
    fs = float(fs) if fs else None
    timing: dict = {}
    history = list(ops.get("processing_history") or [])

    # 2) registration
    _progress("registration", f"Registering plane {plane}")
    moco, shifts, template, timing["registration"] = _stage_registration(
        raw, reg, runtime, plane_dir, device, logger
    )
    if timing["registration"]:
        history.append(
            _history_entry(
                "masknmf_registration", timing["registration"], strategy=reg.strategy
            )
        )

    # 3) compression
    _progress("compression", f"Compressing plane {plane}")
    mask = _shift_mask(shifts, raw.shape[1:], runtime.exclude_border_radius)
    pmd, timing["compression"] = _stage_compression(
        moco, comp, runtime, plane_dir, device, mask, fs, logger
    )
    if timing["compression"]:
        history.append(
            _history_entry(
                "masknmf_compression",
                timing["compression"],
                rank=int(getattr(pmd, "pmd_rank", 0) or 0),
            )
        )

    # 4) demixing
    _progress("demixing", f"Demixing plane {plane}")
    results, timing["detection"] = _stage_demixing(
        pmd, demix, runtime, plane_dir, device, fs, logger
    )
    if timing["detection"]:
        history.append(_history_entry("masknmf_demixing", timing["detection"]))

    # 5) exports: registered binary + summary images + suite2p sidecars
    _progress("exports", f"Writing outputs for plane {plane}")
    is_registered = shifts is not None
    if runtime.keep_bin and is_registered and (
        force_bin or not (plane_dir / "data.bin").exists()
    ):
        mean_img, max_proj = _write_registered_bin(moco, plane_dir, logger)
    else:
        mean_img, max_proj = _movie_stats(moco)

    n_rois = 0
    if results is not None:
        indices, values, baseline = _extract_footprints(results)
        c = np.asarray(results.ac_array.export_c(), dtype=np.float32)
        info = _outputs.write_plane_outputs(
            plane_dir,
            indices=indices,
            values=values,
            c=c,
            shape=raw.shape[1:],
            baseline=baseline,
        )
        n_rois = info["n_rois"]
        logger.info(f"masknmf: plane {plane} -> {n_rois} ROIs")

    vcorr = None
    if results is not None:
        try:
            ci = np.asarray(results.standard_correlation_images[:])
            vcorr = ci.max(axis=0) if ci.ndim == 3 else ci
        except Exception:
            vcorr = None

    timing["total_plane_runtime"] = time.time() - total_t0
    updates = {
        "Ly": raw.shape[1],
        "Lx": raw.shape[2],
        "nframes": raw.shape[0],
        "plane": plane,
        "nplanes": 1,
        "nchannels": 1,
        "anatomical_only": 0,
        "meanImg": mean_img,
        "max_proj": max_proj,
        "yrange": [0, raw.shape[1]],
        "xrange": [0, raw.shape[2]],
        "timing": {
            "registration": timing.get("registration", 0.0),
            "detection": timing.get("detection", 0.0),
            "extraction": timing.get("compression", 0.0),
            "classification": 0.0,
            "deconvolution": 0.0,
            "total_plane_runtime": timing["total_plane_runtime"],
        },
        "n_rois": n_rois,
        "pipeline": "masknmf",
        "masknmf": settings.to_dict(),
        "processing_history": history,
    }
    if fs:
        updates["fs"] = fs
    if template is not None:
        updates["refImg"] = template
    if vcorr is not None:
        updates["Vcorr"] = vcorr
    ops = _outputs.merge_ops(plane_dir, updates)

    # 6) QC figures — regenerated on every run (idempotent replot)
    if replot:
        _progress("figures", f"Plotting QC for plane {plane}")
        from mbo_utilities.masknmf import qc

        qc.plot_plane_figures(
            plane_dir, shifts=shifts, pmd=pmd, results=results, fs=fs, logger=logger
        )

    # drop the raw binary only when a registered data.bin replaces it —
    # otherwise the plane dir would hold no movie for Suite2pArray to read
    if not runtime.keep_raw and (plane_dir / "data.bin").exists():
        del moco
        if hasattr(raw, "_mmap"):
            raw._mmap.close()
        del raw
        try:
            (plane_dir / "data_raw.bin").unlink()
        except OSError:
            pass

    return plane_dir / "ops.npy"


def run_volume(
    input_data,
    save_path,
    planes: list[int] | None = None,
    settings: MasknmfSettings | dict | None = None,
    metadata: dict | None = None,
    frame_indices: list[int] | None = None,
    channel: int | None = None,
    replot: bool = True,
    writer_kwargs: dict | None = None,
    logger=None,
    progress_callback=None,
) -> list[Path]:
    """Run the masknmf pipeline over selected 1-based z-planes."""
    logger = logger or log.get()
    from mbo_utilities import imread

    arr = input_data if hasattr(input_data, "shape") else imread(input_data)
    if planes is None:
        nz = int(arr._shape5d()[2]) if hasattr(arr, "_shape5d") else 1
        planes = list(range(1, nz + 1))

    ops_files: list[Path] = []
    for i, plane in enumerate(planes):
        def _plane_progress(step="", message="", _i=i):
            if progress_callback is not None:
                progress_callback(
                    plane=_i, total_planes=len(planes), step=step, message=message
                )

        ops_files.append(
            run_plane(
                arr,
                save_path,
                plane=plane,
                settings=settings,
                metadata=metadata,
                frame_indices=frame_indices,
                channel=channel,
                replot=replot,
                writer_kwargs=writer_kwargs,
                logger=logger,
                progress_callback=_plane_progress,
            )
        )

    if replot and len(ops_files) > 1:
        from mbo_utilities.masknmf import qc

        qc.plot_volume_figures(Path(save_path), ops_files, logger=logger)
    return ops_files
