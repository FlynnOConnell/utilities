"""Optional-dependency status for the launcher table and ``mbo --check-install``.

Each feature reports what it is for, whether it works on this machine, and
the command that fixes it. GPU checks compare the wheel against the driver's
CUDA version and the card's compute capability, not just ``is_available()``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import version as _dist_version


def _check_import(module_name: str) -> bool:
    """True when ``module_name`` is importable, without importing it."""
    return importlib.util.find_spec(module_name) is not None


def _get_cached_flag(key: str, fallback_check) -> bool:
    """Cached availability flag, else ``fallback_check()``."""
    try:
        from mbo_utilities.env_cache import get_cached_packages

        cached = get_cached_packages()
        if cached and key in cached:
            return cached[key].get("available", False)
    except Exception:
        pass
    return fallback_check()


HAS_SUITE2P: bool = _get_cached_flag(
    "suite2p", lambda: _check_import("lbm_suite2p_python") and _check_import("suite2p")
)
HAS_CUPY: bool = _get_cached_flag("cupy", lambda: _check_import("cupy"))
HAS_TORCH: bool = _get_cached_flag("torch", lambda: _check_import("torch"))
HAS_RASTERMAP: bool = _get_cached_flag("rastermap", lambda: _check_import("rastermap"))
HAS_MASKNMF: bool = _get_cached_flag("masknmf", lambda: _check_import("masknmf"))
HAS_IMGUI: bool = _get_cached_flag("imgui_bundle", lambda: _check_import("imgui_bundle"))
HAS_FASTPLOTLIB: bool = _get_cached_flag("fastplotlib", lambda: _check_import("fastplotlib"))
HAS_PYQT6: bool = _get_cached_flag("pyqt6", lambda: _check_import("PyQt6"))
HAS_NAPARI: bool = _get_cached_flag("napari", lambda: _check_import("napari"))
HAS_NAPARI_OME_ZARR: bool = _get_cached_flag(
    "napari_ome_zarr", lambda: _check_import("napari_ome_zarr")
)
HAS_NAPARI_ANIMATION: bool = _get_cached_flag(
    "napari_animation", lambda: _check_import("napari_animation")
)


class Status(Enum):
    """Health of one feature."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    MISSING = "missing"


@dataclass
class FeatureStatus:
    """One package: what it is for, whether it works, and how to fix it.

    Parameters
    ----------
    gpu_ok : bool or None
        None for packages that never touch a GPU.
    purpose : str
        What the GUI uses it for; one line.
    hint : str
        Install command that fixes a missing or broken state.
    """

    name: str
    status: Status
    version: str = ""
    message: str = ""
    gpu_ok: bool | None = None
    purpose: str = ""
    hint: str = ""


@dataclass
class CudaInfo:
    """What the driver, PyTorch and CuPy each think CUDA is."""

    driver_version: str | None = None
    pytorch_cuda: str | None = None
    cupy_cuda: str | None = None
    device_name: str | None = None
    device_count: int = 0
    capability: str | None = None


@dataclass
class InstallStatus:
    """Everything ``check_installation`` found."""

    mbo_version: str = ""
    python_version: str = ""
    cuda_info: CudaInfo = field(default_factory=CudaInfo)
    features: list[FeatureStatus] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """True when every installed feature works."""
        return all(f.status in (Status.OK, Status.MISSING) for f in self.features)

    def feature(self, name: str) -> FeatureStatus | None:
        """The feature called ``name``, or None."""
        return next((f for f in self.features if f.name == name), None)


# ---------------------------------------------------------------------------
# wheels that match this machine
# ---------------------------------------------------------------------------

_TORCH_INDEX = "https://download.pytorch.org/whl/"
_SUITE2P_HINT = "uv pip install lbm-suite2p-python suite2p rastermap --no-deps"
_MASKNMF_HINT = "uv pip install git+https://github.com/apasarkar/masknmf-toolbox.git"


def _major(ver: str | None) -> int | None:
    """Integer major of "12.4"-style versions, else None."""
    try:
        return int(str(ver).split(".")[0])
    except (ValueError, IndexError, TypeError):
        return None


def _cc(capability: str | None) -> float | None:
    """Compute capability "6.1" as 6.1, else None."""
    try:
        return float(capability) if capability else None
    except ValueError:
        return None


def recommended_cupy_package(driver_cuda: str | None = None) -> str:
    """CuPy wheel for the driver's CUDA major: cupy-cuda11x, 12x or 13x."""
    if driver_cuda is None:
        from mbo_utilities.gpu import driver_cuda as _driver

        driver_cuda = _driver()
    major = _major(driver_cuda)
    if major is None:
        return "cupy-cuda12x"
    return f"cupy-cuda{min(max(major, 11), 13)}x"


def cupy_install_hint(driver_cuda: str | None = None) -> str:
    """CuPy plus the pip-managed NVRTC and runtime its kernels compile against."""
    pkg = recommended_cupy_package(driver_cuda)
    major = pkg.removeprefix("cupy-cuda").removesuffix("x")
    return f"uv pip install {pkg} nvidia-cuda-nvrtc-cu{major} nvidia-cuda-runtime-cu{major}"


def recommended_torch_tag(driver_cuda: str | None, capability: str | None = None) -> str:
    """PyTorch wheel tag the driver and card can run: cu118, cu126, cu128 or cu130.

    CUDA 13 wheels drop Pascal and Volta (compute capability < 7.5); Blackwell
    (12.x) needs cu128 or later; a pre-12 driver only runs cu118.
    """
    major, cc = _major(driver_cuda), _cc(capability)
    if cc is not None and cc >= 12.0:
        return "cu128"
    if major is None or major < 12:
        return "cu118"
    if major >= 13 and (cc is None or cc >= 7.5):
        return "cu130"
    return "cu126"


def torch_install_hint(driver_cuda: str | None = None, capability: str | None = None) -> str:
    """Install command for the PyTorch wheel this machine can run."""
    return f"uv pip install torch --index-url {_TORCH_INDEX}{recommended_torch_tag(driver_cuda, capability)}"


def _arch_supported(arches: list[str], capability: tuple[int, int]) -> bool:
    """True when the wheel ships a kernel (sm_XY) or PTX (compute_XY <= card) for the card."""
    want = capability[0] * 10 + capability[1]
    for arch in arches:
        kind, _, num = arch.partition("_")
        if not num.isdigit():
            continue
        if (kind == "sm" and int(num) == want) or (kind == "compute" and int(num) <= want):
            return True
    return not arches


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

_TORCH_PURPOSE = "suite2p registration, cellpose and masknmf run on its device"
_CUPY_PURPOSE = "z-registration of ScanImage tiffs only (imwrite register_z); numpy otherwise"


def _check_pytorch(
    driver_cuda: str | None, capability: str | None = None
) -> tuple[FeatureStatus, str | None, str | None]:
    """PyTorch and whether its CUDA build drives this card; returns (status, build, capability).

    ``capability`` is the card's compute capability from the driver, so the
    hint fits the card even when the installed wheel cannot see it.
    """
    hint = torch_install_hint(driver_cuda, capability)

    def _feat(status, ver="", msg="", gpu_ok=None, hint=hint):
        return FeatureStatus("PyTorch", status, ver, msg, gpu_ok, _TORCH_PURPOSE, hint)

    if not _check_import("torch"):
        return _feat(Status.MISSING, msg="not installed"), None, None
    try:
        import torch
    except Exception as e:
        return _feat(Status.ERROR, msg=f"import failed: {str(e)[:60]}"), None, None
    ver = torch.__version__
    build = getattr(torch.version, "cuda", None)
    if not build:
        return _feat(Status.WARN, ver, "CPU-only build", gpu_ok=False), None, None
    if not torch.cuda.is_available():
        if driver_cuda is None:
            msg = f"CUDA {build} build, no NVIDIA driver"
        elif (_major(driver_cuda) or 0) < (_major(build) or 0):
            msg = f"driver supports CUDA {driver_cuda}, wheel needs {build}"
        else:
            msg = f"CUDA {build} build, no usable GPU"
        return _feat(Status.WARN, ver, msg, gpu_ok=False), build, None
    try:
        cap = tuple(torch.cuda.get_device_capability(0))
        name = torch.cuda.get_device_name(0)
        arches = list(torch.cuda.get_arch_list())
    except Exception as e:
        return _feat(Status.ERROR, ver, f"CUDA init failed: {str(e)[:60]}", gpu_ok=False), build, None
    capability = f"{cap[0]}.{cap[1]}"
    hint = torch_install_hint(driver_cuda, capability)
    if not _arch_supported(arches, cap):
        msg = f"{name} (sm_{cap[0]}{cap[1]}) has no kernels in the CUDA {build} wheel"
        return _feat(Status.ERROR, ver, msg, gpu_ok=False, hint=hint), build, capability
    return _feat(Status.OK, ver, f"CUDA {build}, {name}", gpu_ok=True, hint=hint), build, capability


def _check_cupy(driver_cuda: str | None) -> tuple[FeatureStatus, str | None]:
    """CuPy, its CUDA runtime, and whether NVRTC can compile a kernel; returns (status, runtime)."""
    hint = cupy_install_hint(driver_cuda)

    def _feat(status, ver="", msg="", gpu_ok=None):
        return FeatureStatus("CuPy", status, ver, msg, gpu_ok, _CUPY_PURPOSE, hint)

    if not _check_import("cupy"):
        return _feat(Status.MISSING, msg="not installed"), None
    try:
        import cupy as cp
    except Exception as e:
        return _feat(Status.ERROR, msg=f"import failed: {str(e)[:60]}"), None
    ver = cp.__version__
    try:
        cp.array([1, 2, 3])
        rt = cp.cuda.runtime.runtimeGetVersion()
        runtime = f"{rt // 1000}.{(rt % 1000) // 10}"
    except Exception as e:
        return _feat(Status.ERROR, ver, f"CUDA init failed: {str(e)[:60]}", gpu_ok=False), None
    drv, cur = _major(driver_cuda), _major(runtime)
    if drv is not None and cur is not None and drv < cur:
        msg = f"built for CUDA {runtime}, driver supports {driver_cuda}"
        return _feat(Status.ERROR, ver, msg, gpu_ok=False), runtime
    try:
        kernel = cp.ElementwiseKernel("float32 x", "float32 y", "y = x * 2", "mbo_probe")
        kernel(cp.array([1.0], dtype="float32"), cp.empty(1, dtype="float32"))
    except Exception:
        return _feat(Status.ERROR, ver, "NVRTC missing, kernels cannot compile", gpu_ok=False), runtime
    return _feat(Status.OK, ver, f"CUDA {runtime}", gpu_ok=True), runtime


def _check_pkg(
    import_name: str, dist_name: str, display: str, purpose: str, hint: str
) -> FeatureStatus:
    """Version by metadata, so noisy packages (suite2p's pynwb warning) never import."""
    if not _check_import(import_name):
        return FeatureStatus(display, Status.MISSING, "", "not installed", None, purpose, hint)
    try:
        ver = _dist_version(dist_name)
    except Exception:
        ver = ""
    return FeatureStatus(display, Status.OK, ver, "ready", None, purpose, hint)


def _on_torch(feat: FeatureStatus, torch: FeatureStatus) -> FeatureStatus:
    """Mark a package that computes on the PyTorch device with PyTorch's GPU state."""
    if feat.status is not Status.OK:
        return feat
    feat.gpu_ok = torch.gpu_ok
    if torch.status is Status.MISSING:
        feat.status, feat.message, feat.hint = Status.WARN, "PyTorch not installed", torch.hint
    elif not torch.gpu_ok:
        feat.status, feat.message, feat.hint = Status.WARN, f"CPU: {torch.message}", torch.hint
    return feat


def check_installation(callback=None) -> InstallStatus:
    """Probe every optional dependency; ``callback(progress, message)`` reports steps."""

    def _update(p: float, msg: str):
        if callback:
            with contextlib.suppress(Exception):
                callback(p, msg)

    from mbo_utilities.gpu import driver_cuda, gpu_devices

    status = InstallStatus()
    try:
        import mbo_utilities

        status.mbo_version = getattr(mbo_utilities, "__version__", "unknown")
    except ImportError:
        status.mbo_version = "not installed"
    status.python_version = ".".join(str(v) for v in sys.version_info[:3])

    _update(0.1, "Checking NVIDIA driver...")
    cuda = status.cuda_info
    cuda.driver_version = driver_cuda()
    devices = gpu_devices()
    if devices:
        cuda.device_name, cuda.device_count = devices[0]["name"], len(devices)
        cuda.capability = devices[0].get("compute_cap")

    _update(0.3, "Checking PyTorch...")
    torch, cuda.pytorch_cuda, seen = _check_pytorch(cuda.driver_version, cuda.capability)
    cuda.capability = seen or cuda.capability
    status.features.append(torch)

    _update(0.5, "Checking CuPy...")
    cupy, cuda.cupy_cuda = _check_cupy(cuda.driver_version)
    status.features.append(cupy)

    _update(0.7, "Checking pipelines...")
    status.features += [
        _on_torch(
            _check_pkg(
                "lbm_suite2p_python", "lbm-suite2p-python", "LBM-Suite2p-Python",
                "suite2p pipeline; registration on the PyTorch device", _SUITE2P_HINT,
            ),
            torch,
        ),
        _check_pkg("suite2p", "suite2p", "Suite2p", "core of LBM-Suite2p-Python", _SUITE2P_HINT),
        _on_torch(
            _check_pkg(
                "cellpose", "cellpose", "Cellpose",
                "anatomical detection in suite2p, on the PyTorch device", "uv pip install cellpose",
            ),
            torch,
        ),
        _on_torch(
            _check_pkg(
                "masknmf", "masknmf", "MaskNMF",
                "masknmf pipeline and curation GUI, on the PyTorch device", _MASKNMF_HINT,
            ),
            torch,
        ),
        _check_pkg("rastermap", "rastermap", "Rastermap", "sorts suite2p traces", _SUITE2P_HINT),
    ]

    _update(0.9, "Checking napari...")
    napari_hint = "uv pip install 'mbo_utilities[napari]'"
    napari = _check_pkg("napari", "napari", "Napari", "the napari viewer mode", napari_hint)
    status.features.append(napari)
    if napari.status is Status.OK:
        status.features += [
            _check_pkg("napari_ome_zarr", "napari-ome-zarr", "napari-ome-zarr", "zarr in napari", napari_hint),
            _check_pkg("napari_animation", "napari-animation", "napari-animation", "movies from napari", napari_hint),
        ]
    _update(1.0, "Done")
    return status


def gpu_summary(cuda: CudaInfo) -> str:
    """One line: card and driver CUDA, or why there is none."""
    if cuda.driver_version is None:
        return "no NVIDIA driver"
    card = cuda.device_name or "NVIDIA GPU"
    if cuda.device_count > 1:
        card += f" (+{cuda.device_count - 1})"
    cc = f" sm {cuda.capability}" if cuda.capability else ""
    return f"{card}{cc}, driver CUDA {cuda.driver_version}"


def print_status_cli(status: InstallStatus):
    """Print the status the way the launcher table shows it."""
    import click

    click.echo(f"\nmbo_utilities v{status.mbo_version} | Python {status.python_version}")
    click.echo(f"GPU: {gpu_summary(status.cuda_info)}")
    click.echo("=" * 60)
    marks = {
        Status.OK: ("[OK]", "green"),
        Status.WARN: ("[! ]", "yellow"),
        Status.ERROR: ("[X ]", "red"),
        Status.MISSING: ("[ -]", "bright_black"),
    }
    for f in status.features:
        mark, color = marks[f.status]
        ver = f" {f.version}" if f.version else ""
        dev = {True: "  GPU", False: "  CPU"}.get(f.gpu_ok, "")
        detail = "" if f.message in ("", "ready") else f"  {f.message}"
        click.echo(f"  {click.style(mark, fg=color)} {click.style(f.name + ver, fg=color)}{dev}{detail}")
        click.echo(click.style(f"       {f.purpose}", fg="bright_black"))
        if f.status is not Status.OK and f.hint:
            click.echo(click.style(f"       fix: {f.hint}", fg="cyan"))
    click.echo("")
    if status.all_ok:
        click.secho("Installation OK", fg="green", bold=True)
    else:
        click.secho("Issues detected - see above", fg="yellow")
