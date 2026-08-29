"""One environment definition, built the same way here, on a worker, or anywhere else.

A pipeline needs an interpreter the lab packages actually support, those
packages from the branches they live on, and a torch matching whatever GPU the
box has. Hard-coding any of that into the worker script means the environment
only exists on Google Compute Engine; keeping it here means the same two lines
build it on a laptop, on a SLURM node, or on the next machine this one deploys.

``uv`` does the work: it fetches the interpreter (the Deep Learning VM image
ships 3.10, which most of these packages refuse), and ``--torch-backend auto``
reads the driver and picks the matching CUDA wheels instead of guessing.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DIR_VENV_DEFAULT = "/opt/imgui-cloud-venv"
PYTHON_DEFAULT = "3.12"
BACKEND_TORCH_DEFAULT = "auto"
URL_INSTALL_UV = "https://astral.sh/uv/install.sh"
TIMEOUT_BUILD_S = 1800.0


@dataclass
class EnvSpec:
    """
    What one environment is: an interpreter, requirements, and a torch build.

    Parameters
    ----------
    python : str
        Version uv fetches and builds the environment with.
    requirements : list of str
        Anything ``uv pip install`` accepts, including ``git+https://...@branch``.
    torch_backend : str
        Passed to ``uv pip install --torch-backend``; ``auto`` reads the GPU
        driver on the machine being built and picks the matching CUDA wheels.
    dir_venv : str
        Where the environment is created.
    """

    python: str = PYTHON_DEFAULT
    requirements: list = field(default_factory=list)
    torch_backend: str = BACKEND_TORCH_DEFAULT
    dir_venv: str = DIR_VENV_DEFAULT

    @property
    def filepath_python(self) -> str:
        """The interpreter this environment provides once it is built."""
        if os.name == "nt":
            return str(Path(self.dir_venv) / "Scripts" / "python.exe")
        return str(Path(self.dir_venv) / "bin" / "python")


def script_bootstrap(
    spec: EnvSpec, install_uv: bool = True, standalone: bool = False
) -> str:
    """
    Bash that builds ``spec`` from nothing, for a worker or any other server.

    Leaves the interpreter path in ``$PY``. ``standalone`` adds the shell
    preamble and the ``fail`` helper that the worker script already defines,
    so the output can be run on its own on another machine.
    """
    quoted = " ".join(shlex.quote(r) for r in spec.requirements)
    lines = []
    if standalone:
        lines += [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            'fail() { echo "[env] FAILED: $1" >&2; exit 1; }',
        ]
    if install_uv:
        lines += [
            'echo "[env] installing uv"',
            "export UV_INSTALL_DIR=/usr/local/bin",
            "export UV_LINK_MODE=copy",
            f'curl -LsSf {URL_INSTALL_UV} | sh || fail "could not install uv"',
            'export PATH="/usr/local/bin:$PATH"',
        ]
    lines += [
        f'echo "[env] python {spec.python} in {spec.dir_venv}"',
        f'uv venv --python {spec.python} {spec.dir_venv} || fail "uv venv failed"',
        f"PY={shlex.quote(spec.filepath_python)}",
        'echo "[env] $($PY --version 2>&1)"',
    ]
    if quoted:
        lines += [
            f"""echo "[env] installing: {quoted.replace('"', "'")}\"""",
            f'uv pip install --python "$PY" --torch-backend {spec.torch_backend} '
            f'{quoted} || fail "pip install failed"',
        ]
    return "\n".join(lines)


def path_uv() -> str:
    """Path of the ``uv`` executable, empty when it is not installed here."""
    return shutil.which("uv") or ""


def build_here(spec: EnvSpec, timeout: float = TIMEOUT_BUILD_S) -> str:
    """
    Build ``spec`` on this machine and return the interpreter it produced.

    Raises
    ------
    RuntimeError
        If uv is missing, or either step fails; the message is uv's own.
    """
    uv = path_uv()
    if not uv:
        raise RuntimeError(f"uv is not installed: see {URL_INSTALL_UV}")
    run_checked([uv, "venv", "--python", spec.python, spec.dir_venv], timeout=timeout)
    if spec.requirements:
        run_checked(
            [
                uv,
                "pip",
                "install",
                "--python",
                spec.filepath_python,
                "--torch-backend",
                spec.torch_backend,
                *spec.requirements,
            ],
            timeout=timeout,
        )
    return spec.filepath_python


def run_checked(command: list, timeout: float) -> str:
    """Run ``command``, returning stdout and raising its stderr on failure."""
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        message = (done.stderr or done.stdout).strip()
        raise RuntimeError(message or f"{command[0]} failed: {' '.join(command[1:])}")
    return done.stdout


def spec_for(pipeline: str, python: str = "", dir_venv: str = "") -> EnvSpec:
    """The environment one registered pipeline needs."""
    from imgui_cloud import pipelines

    spec = pipelines.get(pipeline)
    return EnvSpec(
        python=python or PYTHON_DEFAULT,
        requirements=list(spec.pip),
        dir_venv=dir_venv or str(Path(sys.prefix).parent / f"venv-{pipeline}"),
    )
