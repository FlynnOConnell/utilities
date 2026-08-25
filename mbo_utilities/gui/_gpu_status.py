"""GPU status indicator for pipeline tabs.

Probes whether torch can actually launch kernels on the device the current
pipeline is configured to use. The probe runs in a subprocess so the GUI
process never imports torch or holds a CUDA context; results are cached per
device string for the session.
"""

import json
import subprocess
import sys
import threading

from imgui_bundle import imgui

_PROBE = """\
import json, sys
dev = sys.argv[1]
try:
    import torch
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            out = {"ok": False, "use": "cpu",
                   "reason": "CUDA is not available in this torch build"}
        else:
            name = torch.cuda.get_device_name(0)
            try:
                (torch.ones(2, device=dev) * 2).sum().item()
                out = {"ok": True, "use": dev, "name": name}
            except Exception:
                cc = "sm_%d%d" % torch.cuda.get_device_capability(0)
                archs = ", ".join(torch.cuda.get_arch_list())
                out = {"ok": False, "use": "cpu", "name": name,
                       "reason": f"this torch build has no kernels for {name} "
                                 f"({cc}; build supports {archs})"}
    else:
        out = {"ok": True, "use": "cpu"}
except Exception as e:
    out = {"ok": False, "use": "cpu",
           "reason": str(e).splitlines()[0] if str(e) else type(e).__name__}
print(json.dumps(out))
"""

_INSTALL_HINT = (
    "Reinstall torch for this GPU, e.g.: uv pip install torch torchvision "
    "--index-url https://download.pytorch.org/whl/cu126 "
    "(pre-Turing GPUs need cu126 or cu118, not cu13x)."
)

_results: dict[str, dict] = {}
_pending: set[str] = set()
_lock = threading.Lock()

_GREEN = imgui.ImVec4(0.3, 0.85, 0.3, 1.0)
_YELLOW = imgui.ImVec4(0.95, 0.85, 0.25, 1.0)
_RED = imgui.ImVec4(0.95, 0.35, 0.35, 1.0)


def _probe(device: str) -> None:
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PROBE, device],
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        out = {"ok": False, "use": "cpu", "reason": f"probe failed: {e}"}
    with _lock:
        _results[device] = out
        _pending.discard(device)


def get_status(device: str) -> dict | None:
    """Cached probe result for ``device``; starts a background probe and
    returns None while it is still running."""
    with _lock:
        if device in _results:
            return _results[device]
        if device not in _pending:
            _pending.add(device)
            threading.Thread(
                target=_probe, args=(device,), daemon=True, name="gpu-probe"
            ).start()
    return None


def draw_gpu_status(device: str = "auto") -> None:
    """Colored dot + the device the analysis will run on, with a tooltip
    holding the validation result."""
    device = device or "auto"
    status = get_status(device)
    if status is None:
        color, label, tip = _YELLOW, f"{device} (validating...)", "validating"
    elif status["ok"]:
        color = _GREEN
        label = status.get("name") or status["use"]
        tip = "torch is properly configured"
    else:
        color = _RED
        name = status.get("name") or device
        reason = status.get("reason", "unusable")
        if device.startswith("cuda"):
            label = f"{name} (unusable)"
            tip = (
                f"{reason}. Device is set to '{device}' so the run will fail; "
                f"set device to Auto/CPU or fix torch. {_INSTALL_HINT}"
            )
        else:
            label = f"{name} -> cpu"
            tip = f"{reason}. Falling back to cpu (much slower). {_INSTALL_HINT}"

    imgui.begin_group()
    imgui.push_style_color(imgui.Col_.text, color)
    imgui.bullet()
    imgui.pop_style_color()
    imgui.same_line()
    imgui.text(f"Device: {label}")
    imgui.end_group()
    if imgui.is_item_hovered():
        imgui.set_tooltip(tip)
