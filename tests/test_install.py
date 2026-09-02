"""The dependency checker behind the launcher table and ``mbo --check-install``.

PyTorch is judged against the driver's CUDA and the card's compute
capability, every feature carries a purpose and a fix, and the cache round
trips the dataclasses. Torch and cupy are faked in ``sys.modules``: the
checker only reads a few attributes, and the real imports are slow.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mbo_utilities import install
from mbo_utilities.install import FeatureStatus, Status


# ----------------------------------------------------------------------
# picking a wheel
# ----------------------------------------------------------------------


class TestRecommendedTorchTag:
    @pytest.mark.parametrize(
        ("driver", "cc", "tag"),
        [
            (None, None, "cu118"),
            ("11.8", None, "cu118"),
            ("12.4", None, "cu126"),
            ("12.4", "6.1", "cu126"),
            ("13.0", "6.1", "cu126"),  # Pascal has no cu130 kernels
            ("13.0", "8.6", "cu130"),
            ("13.0", None, "cu130"),
            ("12.8", "12.0", "cu128"),  # Blackwell
        ],
    )
    def test_matches_driver_and_card(self, driver, cc, tag):
        assert install.recommended_torch_tag(driver, cc) == tag
        assert install.torch_install_hint(driver, cc).endswith(tag)


class TestRecommendedCupy:
    @pytest.mark.parametrize(
        ("driver", "pkg"),
        [("11.4", "cupy-cuda11x"), ("12.4", "cupy-cuda12x"), ("13.0", "cupy-cuda13x"), ("9.0", "cupy-cuda11x")],
    )
    def test_major_picks_the_variant(self, driver, pkg):
        assert install.recommended_cupy_package(driver) == pkg
        assert install.cupy_install_hint(driver).startswith(f"uv pip install {pkg} nvidia-cuda-nvrtc")

    def test_unknown_driver_probes_nvidia_smi(self, monkeypatch):
        from mbo_utilities import gpu

        monkeypatch.setattr(gpu, "driver_cuda", lambda: "13.1")
        assert install.recommended_cupy_package(None) == "cupy-cuda13x"


class TestArchSupported:
    def test_exact_kernel(self):
        assert install._arch_supported(["sm_75", "sm_86", "compute_90"], (8, 6))

    def test_ptx_jit_for_a_newer_card(self):
        assert install._arch_supported(["sm_80", "compute_80"], (8, 9))

    def test_older_card_than_every_entry(self):
        assert not install._arch_supported(["sm_75", "sm_120", "compute_120"], (6, 1))

    def test_no_list_means_no_opinion(self):
        assert install._arch_supported([], (6, 1))


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------


class TestDriverCuda:
    def test_parses_the_banner(self, monkeypatch):
        from mbo_utilities import gpu

        banner = "| NVIDIA-SMI 560.94  Driver Version: 560.94  CUDA Version: 12.6  |\n"
        monkeypatch.setattr(gpu, "_run", lambda cmd, timeout=5: banner)
        assert gpu.driver_cuda() == "12.6"

    def test_no_driver(self, monkeypatch):
        from mbo_utilities import gpu

        monkeypatch.setattr(gpu, "_run", lambda cmd, timeout=5: None)
        assert gpu.driver_cuda() is None

    def test_devices_carry_compute_capability(self, monkeypatch):
        from mbo_utilities import gpu

        row = "0, NVIDIA GeForce GTX 1080 Ti, 11264, 900, 10364, 3, 45, WDDM, WDDM, 6.1\n"
        monkeypatch.setattr(gpu, "_run", lambda cmd, timeout=5: row)
        (dev,) = gpu.gpu_devices()
        assert (dev["name"], dev["compute_cap"], dev["driver_model"]) == (
            "NVIDIA GeForce GTX 1080 Ti", "6.1", "WDDM",
        )


# ----------------------------------------------------------------------
# pytorch against this machine
# ----------------------------------------------------------------------


def _fake_torch(monkeypatch, *, cuda=None, available=False, cap=(8, 6), arches=(), name="RTX 3080"):
    torch = SimpleNamespace(
        __version__="2.8.0" + (f"+cu{cuda.replace('.', '')}" if cuda else "+cpu"),
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_capability=lambda i=0: cap,
            get_device_name=lambda i=0: name,
            get_arch_list=lambda: list(arches),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(install, "_check_import", lambda m: m == "torch")
    return torch


class TestCheckPytorch:
    def test_missing(self, monkeypatch):
        monkeypatch.setattr(install, "_check_import", lambda m: False)
        feat, build, cap = install._check_pytorch("12.4")
        assert (feat.status, build, cap) == (Status.MISSING, None, None)
        assert feat.hint.endswith("cu126") and feat.purpose

    def test_cpu_build_is_a_warning(self, monkeypatch):
        _fake_torch(monkeypatch, cuda=None)
        feat, _, _ = install._check_pytorch("12.4")
        assert feat.status is Status.WARN and feat.gpu_ok is False
        assert feat.message == "CPU-only build"

    def test_driver_older_than_the_wheel(self, monkeypatch):
        _fake_torch(monkeypatch, cuda="13.0", available=False)
        feat, build, _ = install._check_pytorch("12.4")
        assert feat.status is Status.WARN and build == "13.0"
        assert feat.message == "driver supports CUDA 12.4, wheel needs 13.0"
        assert feat.hint.endswith("cu126")

    def test_no_driver_at_all(self, monkeypatch):
        _fake_torch(monkeypatch, cuda="12.6", available=False)
        feat, _, _ = install._check_pytorch(None)
        assert "no NVIDIA driver" in feat.message

    def test_pascal_on_a_cu130_wheel_is_an_error(self, monkeypatch):
        _fake_torch(
            monkeypatch, cuda="13.0", available=True, cap=(6, 1),
            arches=["sm_75", "sm_80", "sm_120", "compute_120"], name="GTX 1080 Ti",
        )
        feat, build, cap = install._check_pytorch("13.0")
        assert feat.status is Status.ERROR and feat.gpu_ok is False
        assert cap == "6.1"
        assert "GTX 1080 Ti (sm_61)" in feat.message
        assert feat.hint.endswith("cu126"), "the fix is the last wheel with Pascal kernels"

    def test_cpu_build_hint_uses_the_cards_capability(self, monkeypatch):
        _fake_torch(monkeypatch, cuda=None)
        feat, _, _ = install._check_pytorch("13.0", "6.1")
        assert feat.hint.endswith("cu126"), "a Pascal card on a CUDA 13 driver still wants cu126"
        assert install._check_pytorch("13.0", None)[0].hint.endswith("cu130")

    def test_working_gpu(self, monkeypatch):
        _fake_torch(monkeypatch, cuda="12.6", available=True, arches=["sm_86", "compute_86"])
        feat, build, cap = install._check_pytorch("12.6")
        assert (feat.status, feat.gpu_ok, build, cap) == (Status.OK, True, "12.6", "8.6")
        assert feat.message == "CUDA 12.6, RTX 3080"


class TestOnTorch:
    """Packages that compute on the PyTorch device inherit its GPU state."""

    @staticmethod
    def _pkg():
        return FeatureStatus("Cellpose", Status.OK, "4.0", "ready", None, "detect", "pip x")

    def test_gpu_torch(self):
        torch = FeatureStatus("PyTorch", Status.OK, gpu_ok=True)
        assert install._on_torch(self._pkg(), torch).gpu_ok is True

    def test_cpu_torch_warns_with_the_reason(self):
        torch = FeatureStatus("PyTorch", Status.WARN, message="CPU-only build", gpu_ok=False)
        feat = install._on_torch(self._pkg(), torch)
        assert (feat.status, feat.gpu_ok, feat.message) == (Status.WARN, False, "CPU: CPU-only build")

    def test_missing_torch(self):
        torch = FeatureStatus("PyTorch", Status.MISSING, hint="pip torch")
        feat = install._on_torch(self._pkg(), torch)
        assert (feat.message, feat.hint) == ("PyTorch not installed", "pip torch")

    def test_cpu_torch_points_at_the_torch_wheel(self):
        torch = FeatureStatus("PyTorch", Status.WARN, message="CPU-only build", gpu_ok=False, hint="pip torch")
        assert install._on_torch(self._pkg(), torch).hint == "pip torch"

    def test_missing_package_is_left_alone(self):
        pkg = FeatureStatus("Cellpose", Status.MISSING)
        torch = FeatureStatus("PyTorch", Status.OK, gpu_ok=True)
        assert install._on_torch(pkg, torch).gpu_ok is None


class TestCheckPkg:
    def test_missing(self, monkeypatch):
        monkeypatch.setattr(install, "_check_import", lambda m: False)
        feat = install._check_pkg("x", "x", "X", "does x", "pip install x")
        assert (feat.status, feat.message, feat.purpose, feat.hint) == (
            Status.MISSING, "not installed", "does x", "pip install x",
        )

    def test_installed_reads_metadata_not_the_module(self, monkeypatch):
        monkeypatch.setattr(install, "_check_import", lambda m: True)
        monkeypatch.setattr(install, "_dist_version", lambda d: "1.2.3")
        feat = install._check_pkg("x", "x-dist", "X", "does x", "")
        assert (feat.status, feat.version) == (Status.OK, "1.2.3")


# ----------------------------------------------------------------------
# the whole report
# ----------------------------------------------------------------------


def _fake_status():
    from mbo_utilities.install import CudaInfo, InstallStatus

    return InstallStatus(
        mbo_version="1.0",
        python_version="3.12.0",
        cuda_info=CudaInfo("12.4", "12.6", None, "GTX 1080 Ti", 1, "6.1"),
        features=[
            FeatureStatus("PyTorch", Status.OK, "2.8.0+cu126", "CUDA 12.6, GTX 1080 Ti", True, "torch", "pip t"),
            FeatureStatus("CuPy", Status.MISSING, "", "not installed", None, "z-reg", "pip c"),
            FeatureStatus("Cellpose", Status.WARN, "4.0", "CPU: x", False, "detect", "pip cp"),
        ],
    )


class TestInstallStatus:
    def test_feature_lookup(self):
        status = _fake_status()
        assert status.feature("CuPy").status is Status.MISSING
        assert status.feature("nope") is None
        assert not status.all_ok

    def test_gpu_summary(self):
        from mbo_utilities.install import CudaInfo, gpu_summary

        assert gpu_summary(_fake_status().cuda_info) == "GTX 1080 Ti sm 6.1, driver CUDA 12.4"
        assert gpu_summary(CudaInfo()) == "no NVIDIA driver"
        two = CudaInfo(driver_version="13.0", device_name="A100", device_count=2)
        assert gpu_summary(two) == "A100 (+1), driver CUDA 13.0"

    def test_cli_print_shows_purpose_and_fix(self, capsys):
        install.print_status_cli(_fake_status())
        out = capsys.readouterr().out
        assert "GPU: GTX 1080 Ti sm 6.1, driver CUDA 12.4" in out
        assert "PyTorch 2.8.0+cu126  GPU" in out
        assert "fix: pip cp" in out and "fix: pip t" not in out
        assert "z-reg" in out

    def test_check_installation_wires_driver_torch_and_packages(self, monkeypatch):
        from mbo_utilities import gpu

        monkeypatch.setattr(gpu, "driver_cuda", lambda: "12.4")
        monkeypatch.setattr(
            gpu, "gpu_devices", lambda: [{"index": 0, "name": "GTX 1080 Ti", "compute_cap": "6.1"}]
        )
        _fake_torch(monkeypatch, cuda="12.6", available=True, cap=(6, 1), arches=["sm_61"])
        monkeypatch.setattr(install, "_check_import", lambda m: m in ("torch", "cellpose"))
        monkeypatch.setattr(install, "_dist_version", lambda d: "9.9")
        status = install.check_installation()
        assert status.cuda_info.device_name == "GTX 1080 Ti"
        assert status.cuda_info.capability == "6.1"
        assert status.feature("PyTorch").gpu_ok is True
        assert status.feature("Cellpose").gpu_ok is True
        assert status.feature("CuPy").status is Status.MISSING
        assert status.feature("MaskNMF").hint.startswith("uv pip install git+")
        assert status.feature("napari-ome-zarr") is None, "plugins only matter with napari"


class TestCacheRoundTrip:
    def test_dataclasses_survive_the_cache(self):
        from mbo_utilities.env_cache import _deserialize_install_status, _serialize_install_status

        status = _fake_status()
        back = _deserialize_install_status(_serialize_install_status(status))
        assert back == status

    def test_old_cache_keys_are_dropped(self):
        from mbo_utilities.env_cache import _deserialize_install_status

        data = {
            "cuda_info": {"nvcc_version": "12.0", "driver_version": "12.4"},
            "features": [{"name": "PyTorch", "status": "ok", "extra": 1}],
        }
        back = _deserialize_install_status(data)
        assert back.cuda_info.driver_version == "12.4"
        assert back.features[0].purpose == ""

    def test_schema_bump_invalidates(self, monkeypatch):
        from mbo_utilities import env_cache

        monkeypatch.setattr(env_cache, "get_env_fingerprint", lambda: "fp")
        from datetime import datetime

        from mbo_utilities import __version__

        base = {"mbo_version": __version__, "env_fingerprint": "fp", "last_updated": datetime.now().isoformat()}
        assert not env_cache.is_cache_valid({**base, "schema": 1})
        assert env_cache.is_cache_valid({**base, "schema": env_cache._CACHE_SCHEMA})


# ----------------------------------------------------------------------
# launcher table
# ----------------------------------------------------------------------


class TestLauncherRows:
    @staticmethod
    def _dialog():
        from mbo_utilities.gui.widgets import file_dialog as fd

        dlg = fd.FileDialog.__new__(fd.FileDialog)
        dlg._install_status = _fake_status()
        dlg._show_deps_popup = False
        return dlg

    def test_rows_follow_table_order_and_skip_unknown(self):
        assert [f.name for f in self._dialog()._dep_rows()] == ["PyTorch", "Cellpose", "CuPy"]

    def test_table_draws_in_a_frame(self):
        from tests.test_manual_roi import _offscreen_selected

        if not _offscreen_selected():
            pytest.skip("needs the offscreen rendercanvas")
        from imgui_bundle import imgui

        from tests.test_imgui_helpers import _draw_in_edge_window

        dlg = self._dialog()
        dlg._show_deps_popup = True

        def body(out):
            dlg._draw_dependency_status_line()
            out.append(imgui.is_popup_open("##deps_popup"))

        drawn = _draw_in_edge_window(340, body)
        assert drawn[0], "the popup opened and the table drew on the first frame"
