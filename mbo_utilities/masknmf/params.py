"""masknmf pipeline parameter dataclasses.

GUI-free and JSON-round-trippable: the pipeline widget serializes these into
worker args, and the runner rebuilds them in the subprocess. Field names for
each stage mirror the upstream ``masknmf.pipelines.configs`` dataclasses so
values pass straight through ``asdict``-style into the strategies.

Stage tri-states follow the suite2p convention: 0=skip, 1=run, 2=force.
"skip" means the stage's cached HDF5 output is required (compression) or the
stage is bypassed entirely (registration); "run" reuses a valid cached output;
"force" always recomputes.
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Any

STAGE_SKIP = 0
STAGE_RUN = 1
STAGE_FORCE = 2

# per-stage native outputs inside a plane dir; gating keys on their presence
MOCO_FILE = "motion_correction.hdf5"
PMD_FILE = "compression.hdf5"
DEMIX_FILE = "demixing_results.hdf5"


@dataclass
class MasknmfRegistrationSettings:
    do_registration: int = 1  # 0=skip, 1=run, 2=force
    strategy: str = "rigid"  # "rigid" | "pwrigid"
    max_shifts: tuple[int, int] = (15, 15)
    # pwrigid only
    num_blocks: tuple[int, int] = (15, 15)
    overlaps: tuple[int, int] = (5, 5)
    max_deviation_rigid: tuple[int, int] = (2, 2)

    def strategy_kwargs(self) -> dict:
        """kwargs for the masknmf motion-corrector constructor."""
        if self.strategy == "pwrigid":
            return {
                "num_blocks": tuple(self.num_blocks),
                "overlaps": tuple(self.overlaps),
                "max_rigid_shifts": tuple(self.max_shifts),
                "max_deviation_rigid": tuple(self.max_deviation_rigid),
            }
        return {"max_shifts": tuple(self.max_shifts)}


@dataclass
class MasknmfCompressionSettings:
    do_compression: int = 1  # 0=skip (requires cached hdf5), 1=run, 2=force
    denoise: bool = True  # CompressDenoiseStrategy vs CompressStrategy
    block_sizes: tuple[int, int] = (20, 20)
    max_components: int = 20
    sim_conf: int = 5
    spatial_avg_factor: int = 1
    temporal_avg_factor: int = 1
    # denoise=True only
    num_epochs: int = 10
    noise_variance_quantile: float = 0.3
    # MaximinSplineDetrend sized from fs (window=40s, knot per 25s)
    detrend: bool = True

    def strategy_kwargs(self) -> dict:
        kw = {
            "block_sizes": tuple(self.block_sizes),
            "max_components": int(self.max_components),
            "sim_conf": int(self.sim_conf),
            "spatial_avg_factor": int(self.spatial_avg_factor),
            "temporal_avg_factor": int(self.temporal_avg_factor),
        }
        if self.denoise:
            kw["num_epochs"] = int(self.num_epochs)
            kw["noise_variance_quantile"] = float(self.noise_variance_quantile)
        return kw


@dataclass
class MasknmfDemixingSettings:
    do_demixing: int = 1  # 0=skip, 1=run, 2=force
    # spatial highpass applied to the PMD movie for initialization
    filter_sigma: float = 4.0
    # superpixel init
    mad_correlation_threshold: float = 0.8
    mad_threshold: float = 1.0
    min_peak_distance: int = 3
    residual_threshold: float = 0.3
    patch_size: tuple[int, int] = (40, 40)
    sign: str = "positive"  # positive | negative | unconstrained
    # NMF (HALS) rounds
    maxiter: int = 40
    support_threshold_hi: float = 0.95
    support_threshold_lo: float = 0.8  # filtered rounds
    unfiltered_support_lo: float = 0.4  # unfiltered rounds
    filtered_passes: int = 2
    unfiltered_passes: int = 3
    ring_radius: int = 10
    background_downsampling_factor: int = 30
    merge_threshold: float = 0.6
    merge_overlap_threshold: float = 0.6
    deletion_threshold: float = 0.2
    min_brightness: float = 1.0
    update_frequency: int = 4
    reassign_background: bool = True

    def init_kwargs(self, detrender: Any = None) -> dict:
        return {
            "mad_correlation_threshold": float(self.mad_correlation_threshold),
            "mad_threshold": float(self.mad_threshold),
            "min_peak_distance": int(self.min_peak_distance),
            "residual_threshold": float(self.residual_threshold),
            "patch_size": tuple(self.patch_size),
            "sign": self.sign,
            "detrender": detrender,
        }

    def nmf_kwargs(self, support_lo: float, ring: bool, detrender: Any = None) -> dict:
        return {
            "maxiter": int(self.maxiter),
            "support_threshold": (float(self.support_threshold_hi), float(support_lo)),
            "deletion_threshold": float(self.deletion_threshold),
            "min_brightness": float(self.min_brightness),
            "ring_model_start_pt": 0 if ring else None,
            "ring_radius": int(self.ring_radius),
            "background_downsampling_factor": int(self.background_downsampling_factor),
            "merge_threshold": float(self.merge_threshold),
            "merge_overlap_threshold": float(self.merge_overlap_threshold),
            "update_frequency": int(self.update_frequency),
            "c_nonneg": True,
            "reassign_background": bool(self.reassign_background),
            "detrender": detrender,
        }


@dataclass
class MasknmfRuntimeSettings:
    # GPU by default: superpixel init needs CUDA, and "auto" quietly fell
    # back to a cpu run that is orders of magnitude slower
    device: str = "cuda"  # auto | cuda | cpu
    frame_batch_size: int = 300
    exclude_border_radius: int = 0
    # native HDF5 stage outputs stay on disk so Skip/Run gating can resume
    keep_intermediates: bool = True
    # suite2p-parity binaries
    keep_bin: bool = True  # write registered data.bin
    keep_raw: bool = False  # keep data_raw.bin after the run


@dataclass
class MasknmfSettings:
    """Complete masknmf pipeline configuration (one worker-args payload)."""

    registration: MasknmfRegistrationSettings = field(
        default_factory=MasknmfRegistrationSettings
    )
    compression: MasknmfCompressionSettings = field(
        default_factory=MasknmfCompressionSettings
    )
    demixing: MasknmfDemixingSettings = field(default_factory=MasknmfDemixingSettings)
    runtime: MasknmfRuntimeSettings = field(default_factory=MasknmfRuntimeSettings)

    def to_dict(self) -> dict:
        d = asdict(self)
        return _tuples_to_lists(d)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MasknmfSettings":
        d = d or {}
        return cls(
            registration=_load_section(MasknmfRegistrationSettings, d.get("registration")),
            compression=_load_section(MasknmfCompressionSettings, d.get("compression")),
            demixing=_load_section(MasknmfDemixingSettings, d.get("demixing")),
            runtime=_load_section(MasknmfRuntimeSettings, d.get("runtime")),
        )


def _tuples_to_lists(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(v) for v in obj]
    return obj


def _load_section(cls: type, d: dict | None):
    d = d or {}
    valid = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in valid:
            continue
        if isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    return cls(**kwargs)


def stage_action(tri: int, cached: bool) -> str:
    """Resolve a stage tri-state against its cached output.

    Returns "skip" (bypass the stage), "reuse" (load its cached output), or
    "compute". What "skip" means is stage-specific (registration: feed raw
    frames downstream; compression: cached PMD required; demixing: stop
    after compression).
    """
    if tri == STAGE_FORCE:
        return "compute"
    if tri == STAGE_SKIP:
        return "skip"
    return "reuse" if cached else "compute"
