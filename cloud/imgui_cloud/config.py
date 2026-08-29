"""
Declarative config for one cloud run: what to compute, and on what hardware.

Mirrors the layout of ``mbo_utilities.hpc.config`` so the two schedulers read
the same way. Sections map to TOML tables:

  [io]        local input dir, local output dir, run name
  [machine]   instance type, accelerator, disks, spot, lifetime cap
  [job]       which pipeline, extra pip requirements, env, pipeline parameters

Connection settings (project / zone / bucket / credentials) are *not* here -
they live in the :class:`~imgui_cloud.credentials.CloudProfile`, because they
belong to the person and machine, not to the analysis.
"""

from __future__ import annotations

from typing import Any

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# a2/a3/a4/g2 machine types carry their GPUs implicitly: asking for
# a2-highgpu-1g already means one A100 40GB, and adding a guest_accelerators
# block to the instance is rejected by the API. Anything else (n1-*) must ask.
FAMILIES_GPU_IMPLICIT = ("a2-", "a3-", "a4-", "g2-")

PREFIX_QUOTA_SPOT = "PREEMPTIBLE_"


@dataclass(frozen=True)
class GpuOption:
    """
    One box worth offering: the machine, the GPUs on it, and what it costs.

    Parameters
    ----------
    machine_type : str
        Compute Engine machine type, e.g. ``a2-highgpu-1g``.
    gpu : str
        Human name of the accelerator, e.g. ``A100 40GB``.
    count : int
        How many of them the machine carries.
    accelerator_type : str
        Compute Engine accelerator id, e.g. ``nvidia-tesla-a100``.
    metric_quota : str
        Regional quota metric that has to be non-zero to launch it; the spot
        variant is the same name behind ``PREEMPTIBLE_``.
    vcpu, memory_gb : int
        What comes with the machine.
    usd_per_hour : float
        On-demand us-central1 list price for the whole machine, GPUs included.
    """

    machine_type: str
    gpu: str
    count: int
    accelerator_type: str
    metric_quota: str
    vcpu: int
    memory_gb: int
    usd_per_hour: float

    @property
    def label(self) -> str:
        """``2x A100 40GB``-style summary of what you get."""
        return f"{self.count}x {self.gpu}"

    @property
    def metric_quota_spot(self) -> str:
        """The separate preemptible metric, where a project has been granted one."""
        return PREFIX_QUOTA_SPOT + self.metric_quota


# List prices are indicative us-central1 numbers, shown so nobody starts an
# eight-hour run without seeing one. Real billing depends on region, disks,
# egress and committed-use discounts.
# fmt: off
GPUS = (
    GpuOption("a2-highgpu-1g",   "A100 40GB", 1, "nvidia-tesla-a100",  "NVIDIA_A100_GPUS",       12,   85,   3.67),
    GpuOption("a2-highgpu-2g",   "A100 40GB", 2, "nvidia-tesla-a100",  "NVIDIA_A100_GPUS",       24,  170,   7.35),
    GpuOption("a2-highgpu-4g",   "A100 40GB", 4, "nvidia-tesla-a100",  "NVIDIA_A100_GPUS",       48,  340,  14.69),
    GpuOption("a2-ultragpu-1g",  "A100 80GB", 1, "nvidia-a100-80gb",   "NVIDIA_A100_80GB_GPUS",  12,  170,   5.07),
    GpuOption("a2-ultragpu-2g",  "A100 80GB", 2, "nvidia-a100-80gb",   "NVIDIA_A100_80GB_GPUS",  24,  340,  10.14),
    GpuOption("a3-highgpu-8g",   "H100 80GB", 8, "nvidia-h100-80gb",   "NVIDIA_H100_GPUS",      208, 1872,  88.49),
    GpuOption("g2-standard-8",   "L4 24GB",   1, "nvidia-l4",          "NVIDIA_L4_GPUS",          8,   32,   0.85),
    GpuOption("g2-standard-24",  "L4 24GB",   2, "nvidia-l4",          "NVIDIA_L4_GPUS",         24,   96,   2.00),
    GpuOption("n1-standard-8",   "T4 16GB",   1, "nvidia-tesla-t4",    "NVIDIA_T4_GPUS",          8,   30,   0.73),
    GpuOption("n1-standard-8",   "V100 16GB", 1, "nvidia-tesla-v100",  "NVIDIA_V100_GPUS",        8,   30,   2.86),
)
# fmt: on

# Offered before the live API has been asked; the panel replaces these with
# the zones the project actually has the chosen accelerator in.
ZONES_A100_COMMON = [
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
    "us-east1-b",
    "us-west1-b",
    "us-west3-b",
    "europe-west4-a",
    "asia-northeast1-a",
]


def quota_effective(quotas: dict, option: GpuOption, spot: bool) -> tuple:
    """
    The limit that really gates ``option``, and the metric it was read from.

    Preemptible quota is opt-in: most projects have none, and Compute Engine
    then bills spot VMs against the ordinary quota. A zero next to a
    ``PREEMPTIBLE_`` metric therefore says nothing about whether the box can
    start, which is why it is only believed when it is above zero.

    Returns
    -------
    tuple
        ``(limit, metric)``, with a limit of -1 when the region has not been
        read yet.
    """
    limit_regular = quotas.get(option.metric_quota, -1.0)
    if not spot:
        return limit_regular, option.metric_quota
    limit_spot = quotas.get(option.metric_quota_spot, -1.0)
    if limit_spot > 0:
        return limit_spot, option.metric_quota_spot
    return limit_regular, option.metric_quota


def recommend_gpu(
    quotas: dict, zones_by_accelerator: dict, zone: str = "", spot: bool = True
) -> GpuOption | None:
    """
    The cheapest box this project can actually run, or None when none can.

    Prefers something the current zone carries, then anything with quota
    somewhere; the caller offers the zone move.
    """
    affordable = sorted(GPUS, key=lambda option: option.usd_per_hour)
    with_quota = [
        option
        for option in affordable
        if quota_effective(quotas, option, spot)[0] >= option.count
    ]
    here = [
        option
        for option in with_quota
        if zone in zones_by_accelerator.get(option.accelerator_type, [zone])
    ]
    for candidates in (here, with_quota):
        if candidates:
            return candidates[0]
    return None


def gpu_option(machine_type: str, accelerator_type: str = "") -> GpuOption | None:
    """The catalog entry for a machine type, disambiguated by accelerator."""
    matches = [option for option in GPUS if option.machine_type == machine_type]
    if accelerator_type:
        exact = [o for o in matches if o.accelerator_type == accelerator_type]
        if exact:
            return exact[0]
    return matches[0] if matches else None


@dataclass
class IOConfig:
    """Local input/output and the run's human-readable name."""

    input: str = ""
    output: str = ""
    name: str = "run"
    dated_subfolder: bool = True


@dataclass
class MachineConfig:
    """
    The box: what gets created, how big its scratch disk is, when it dies.

    ``max_runtime_min`` is the safety net that makes a forgotten run cheap: it
    is set on the instance itself as ``scheduling.max_run_duration`` with a
    termination action of DELETE, so the VM removes itself even if this laptop
    is closed mid-run.
    """

    machine_type: str = "a2-highgpu-1g"
    accelerator_type: str = "nvidia-tesla-a100"
    accelerator_count: int = 1
    spot: bool = True
    boot_disk_gb: int = 200
    boot_disk_type: str = "pd-balanced"
    data_disk_gb: int = 1000
    data_disk_type: str = "pd-ssd"
    data_disk_name: str = ""
    keep_data_disk: bool = False
    image_family: str = "pytorch-latest-gpu"
    image_project: str = "deeplearning-platform-release"
    max_runtime_min: int = 480
    keep_instance: bool = False

    @property
    def needs_guest_accelerator(self) -> bool:
        """True when the machine type does not include its GPUs implicitly."""
        return not self.machine_type.startswith(FAMILIES_GPU_IMPLICIT)

    def cost_per_hour_estimate(self) -> float:
        """
        Rough us-central1 list price in USD/hour, spot-adjusted, disks included.

        Indicative only - real billing depends on region, disks, egress and
        committed-use discounts. Shown in the panel so nobody starts an 8-hour
        run without seeing a number.
        """
        option = gpu_option(self.machine_type, self.accelerator_type)
        rate = option.usd_per_hour if option else self._rate_unlisted()
        disk = (self.boot_disk_gb + self.data_disk_gb) * 0.00023
        return (rate * 0.4 if self.spot else rate) + disk

    def _rate_unlisted(self) -> float:
        """Fallback A100 pricing for a machine type the catalog does not carry."""
        per_gpu = 3.67 if "ultragpu" in self.machine_type else 2.93
        gpus = 1
        suffix = self.machine_type.rsplit("-", 1)[-1]
        if suffix.endswith("g") and suffix[:-1].isdigit():
            gpus = int(suffix[:-1])
        elif self.needs_guest_accelerator:
            gpus = max(1, self.accelerator_count)
        return per_gpu * gpus


@dataclass
class JobConfig:
    """
    The work: which pipeline the worker runs, and with what.

    ``pipeline`` names an entry in :mod:`imgui_cloud.pipelines`. ``pip`` and
    ``env`` extend (not replace) what that pipeline declares, so a run can pin
    a branch without editing the package.
    """

    pipeline: str = "masknmf"
    pip: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    upload_patterns: list = field(default_factory=lambda: ["*"])
    download_patterns: list = field(default_factory=lambda: ["*"])


@dataclass
class CloudConfig:
    """Whole run description: :class:`IOConfig` + :class:`MachineConfig` + :class:`JobConfig`."""

    io: IOConfig = field(default_factory=IOConfig)
    machine: MachineConfig = field(default_factory=MachineConfig)
    job: JobConfig = field(default_factory=JobConfig)

    def validate(self) -> list:
        """Problems that would make this run fail, as human-readable strings."""
        problems = []
        if not self.io.input:
            problems.append("[io] input is empty")
        elif not Path(self.io.input).exists():
            problems.append(f"[io] input does not exist: {self.io.input}")
        if not self.io.output:
            problems.append("[io] output is empty")
        if self.machine.max_runtime_min < 5:
            problems.append("[machine] max_runtime_min must be >= 5")
        if self.machine.data_disk_gb < 10:
            problems.append("[machine] data_disk_gb must be >= 10")
        from imgui_cloud import pipelines

        if self.job.pipeline not in pipelines.available():
            problems.append(
                f"[job] unknown pipeline {self.job.pipeline!r}; "
                f"known: {', '.join(pipelines.available())}"
            )
        return problems

    def to_dict(self) -> dict:
        """Plain nested dict, suitable for JSON or TOML."""
        return asdict(self)


def _coerce(cls, body: dict):
    """Build a dataclass from ``body``, keeping only fields it declares."""
    names_valid = {f.name for f in fields(cls)}
    unknown = set(body) - names_valid
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**{k: v for k, v in body.items() if k in names_valid})


def from_dict(body: dict) -> CloudConfig:
    """Build a :class:`CloudConfig` from a nested dict, rejecting stray keys."""
    return CloudConfig(
        io=_coerce(IOConfig, body.get("io", {})),
        machine=_coerce(MachineConfig, body.get("machine", {})),
        job=_coerce(JobConfig, body.get("job", {})),
    )


def from_toml(filepath_config: str) -> CloudConfig:
    """Load a run config from a TOML file written by :func:`write_template`."""
    with open(filepath_config, "rb") as f:
        return from_dict(tomllib.load(f))


TEMPLATE = """\
# imgui_cloud run config. Connection settings (project, zone, bucket,
# credentials) come from your signed-in profile, not from this file:
#   imgui-cloud login --show

[io]
input = "{input}"          # local folder to upload
output = "{output}"         # local folder results come back to
name = "{name}"
dated_subfolder = true       # write results into <output>/<date>_<name>

[machine]
machine_type = "a2-highgpu-1g"   # 1x A100 40GB
spot = true                      # ~60% cheaper, can be preempted
data_disk_gb = 1000              # scratch disk mounted at /mnt/data
data_disk_type = "pd-ssd"
boot_disk_gb = 200
max_runtime_min = 480            # hard cap: the VM deletes itself at this age
keep_data_disk = false           # true keeps the scratch disk after teardown
keep_instance = false            # true leaves the VM up for debugging

[job]
pipeline = "{pipeline}"
pip = []                         # extra requirements, e.g. ["git+https://..."]

[job.env]
# EXTRA_ENV = "value"

[job.params]
# pipeline keyword arguments, forwarded verbatim
# planes = [1, 2, 3]
"""


def write_template(
    filepath_config: str,
    dir_input: str = "",
    dir_output: str = "",
    name: str = "run",
    pipeline: str = "masknmf",
    overwrite: bool = False,
) -> str:
    """
    Write a commented starter TOML next to the data.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    FileExistsError
        If the file exists and ``overwrite`` is False.
    """
    path = Path(filepath_config)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.write_text(
        TEMPLATE.format(
            input=str(dir_input).replace("\\", "/"),
            output=str(dir_output).replace("\\", "/"),
            name=name,
            pipeline=pipeline,
        )
    )
    return str(path)


def default_config(
    dir_input: str = "", dir_output: str = "", name: str = "run"
) -> CloudConfig:
    """A ready-to-edit config pointed at ``dir_input`` / ``dir_output``."""
    config = CloudConfig()
    config.io.input = str(dir_input)
    config.io.output = str(dir_output)
    config.io.name = name
    return config


def resolve_output_dir(config: CloudConfig) -> Path:
    """``<output>/<date>_<name>``, suffixed ``_2``, ``_3``... if it exists."""
    import datetime

    root = Path(config.io.output or ".")
    if not config.io.dated_subfolder:
        return root
    stamp = datetime.date.today().strftime("%Y_%m_%d")
    base = root / f"{stamp}_{config.io.name}"
    final, n = base, 2
    while final.exists():
        final = base.with_name(f"{base.name}_{n}")
        n += 1
    return final


def merge_overrides(
    config: CloudConfig, overrides: dict[str, Any] | None
) -> CloudConfig:
    """
    Apply ``{"machine.spot": False, "io.name": "x"}``-style overrides in place.

    Raises
    ------
    KeyError
        For a dotted key that names no section or no field.
    """
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section_name, _, field_name = dotted.partition(".")
        section = getattr(config, section_name, None)
        if section is None or not hasattr(section, field_name):
            raise KeyError(f"no such config field: {dotted}")
        setattr(section, field_name, value)
    return config
