"""
The script that actually runs on the A100 box.

Standard library only, and deliberately tiny: the startup script uploads this
file to the worker and runs ``python worker_main.py job.json``. imgui_cloud
itself is never installed on the VM, so a GUI dependency can never break a run.

``job.json`` is what :func:`imgui_cloud.pipelines.build_job` produced::

    {"entry": "mbo_utilities.masknmf:run_volume",
     "kwargs": {"input_data": "/mnt/data/input", "save_path": "/mnt/data/output"}}

Everything printed here lands in the worker log that is streamed back to GCS.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
import traceback


def resolve_entry(entry: str):
    """
    Import ``module:function`` and return the callable.

    Raises
    ------
    ValueError
        If ``entry`` is not of the form ``module:function``.
    """
    module_name, sep, function_name = entry.partition(":")
    if not sep or not module_name or not function_name:
        raise ValueError(f"entry must be 'module:function', got {entry!r}")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def describe_gpus() -> str:
    """One line per visible GPU, or a note that torch could not see any."""
    try:
        import torch
    except ImportError:
        return "torch not installed; cannot report GPUs"
    if not torch.cuda.is_available():
        return "torch reports no CUDA device"
    return "\n".join(
        f"gpu {i}: {torch.cuda.get_device_name(i)} "
        f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB)"
        for i in range(torch.cuda.device_count())
    )


def run_job(job: dict) -> None:
    """Call the job's entry point with its keyword arguments."""
    function = resolve_entry(job["entry"])
    kwargs = job.get("kwargs", {})
    print(f"[worker] calling {job['entry']}", flush=True)
    for key, value in kwargs.items():
        print(f"[worker]   {key} = {value!r}", flush=True)
    result = function(**kwargs)
    print(f"[worker] returned: {result!r}", flush=True)


def main(argv=None) -> int:
    """Entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="imgui_cloud worker")
    parser.add_argument("filepath_job", help="path to job.json")
    args = parser.parse_args(argv)

    with open(args.filepath_job) as f:
        job = json.load(f)

    print(f"[worker] python {sys.version.split()[0]} on {platform.node()}", flush=True)
    print(f"[worker] {describe_gpus()}", flush=True)

    time_start = time.time()
    try:
        run_job(job)
    except Exception:
        traceback.print_exc()
        print(f"[worker] FAILED after {time.time() - time_start:.1f}s", flush=True)
        return 1
    print(f"[worker] OK in {time.time() - time_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
