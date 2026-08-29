"""
What a worker is allowed to run, and how its parameters reach the function.

A :class:`PipelineSpec` is a pure description - a pip requirement list plus a
``module:function`` entry point that takes an input path and an output path.
The worker resolves it with nothing but the standard library (see
:mod:`imgui_cloud.worker_main`), so the VM never installs imgui_cloud itself.

Third-party packages add pipelines without editing this module by declaring an
``imgui_cloud.pipelines`` entry point that resolves to a :class:`PipelineSpec`.
"""

from __future__ import annotations


import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "imgui_cloud.pipelines"


@dataclass
class PipelineSpec:
    """
    One runnable pipeline.

    Parameters
    ----------
    name : str
        Key used in ``[job] pipeline``.
    description : str
        One line shown in the panel and in ``imgui-cloud pipelines``.
    pip : list of str
        Requirements installed on the worker before the entry point is imported.
    entry : str
        ``module:function``. The function is called with the input and output
        paths plus the run parameters.
    arg_input, arg_output : str
        Keyword names the input and output paths are passed under.
    arg_params : str
        Keyword the leftover parameters are bundled into as a dict. Empty means
        they are spread as top-level keyword arguments instead.
    params_flat : list of str
        Parameter keys that stay top-level even when ``arg_params`` is set.
    default_params : dict
        Merged under the run's own ``[job.params]``.
    """

    name: str
    description: str = ""
    pip: list = field(default_factory=list)
    entry: str = ""
    arg_input: str = "input_data"
    arg_output: str = "save_path"
    arg_params: str = ""
    params_flat: list = field(default_factory=list)
    default_params: dict = field(default_factory=dict)


MASKNMF = PipelineSpec(
    name="masknmf",
    description="maskNMF demixing (registration -> compression -> demixing) per z-plane",
    pip=[
        "mbo_utilities",
        "masknmf[multisession] @ git+https://github.com/apasarkar/masknmf-toolbox.git@curation-viz",
    ],
    entry="mbo_utilities.masknmf:run_volume",
    arg_params="settings",
    params_flat=[
        "planes",
        "metadata",
        "frame_indices",
        "channel",
        "replot",
        "writer_kwargs",
    ],
)

SUITE2P = PipelineSpec(
    name="suite2p",
    description="LBM Suite2p: registration, cellpose detection, extraction, dF/F",
    pip=["mbo_utilities", "lbm_suite2p_python"],
    entry="lbm_suite2p_python:run_volume",
    default_params={"keep_reg": False, "keep_raw": False},
)

_BUILTINS = {spec.name: spec for spec in (MASKNMF, SUITE2P)}
_LOADED_ENTRY_POINTS = False


def _load_entry_points() -> None:
    """Merge ``imgui_cloud.pipelines`` entry points into the registry once."""
    global _LOADED_ENTRY_POINTS
    if _LOADED_ENTRY_POINTS:
        return
    _LOADED_ENTRY_POINTS = True

    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            spec = ep.load()
        except Exception:
            logger.exception("could not load pipeline entry point %r", ep.name)
            continue
        if not isinstance(spec, PipelineSpec):
            logger.warning("entry point %r is not a PipelineSpec, skipping", ep.name)
            continue
        _BUILTINS[spec.name] = spec


def available() -> list:
    """Names of every registered pipeline."""
    _load_entry_points()
    return sorted(_BUILTINS)


def get(name: str) -> PipelineSpec:
    """
    Look up a pipeline by name.

    Raises
    ------
    KeyError
        If no pipeline of that name is registered.
    """
    _load_entry_points()
    if name not in _BUILTINS:
        raise KeyError(f"unknown pipeline {name!r}; known: {', '.join(available())}")
    return _BUILTINS[name]


def register(spec: PipelineSpec) -> None:
    """Add or replace a pipeline in the in-process registry."""
    _load_entry_points()
    _BUILTINS[spec.name] = spec


def build_job(
    spec: PipelineSpec,
    dir_input_remote: str,
    dir_output_remote: str,
    params: dict | None = None,
) -> dict:
    """
    Build the ``job.json`` payload the worker executes.

    Parameters
    ----------
    spec : PipelineSpec
        Pipeline to run.
    dir_input_remote, dir_output_remote : str
        Paths *on the worker*, under its mounted data disk.
    params : dict, optional
        Run parameters, merged over ``spec.default_params``.

    Returns
    -------
    dict
        ``{"entry": ..., "kwargs": {...}}`` - everything the worker needs.
    """
    merged = dict(spec.default_params)
    merged.update(params or {})

    kwargs = {spec.arg_input: dir_input_remote, spec.arg_output: dir_output_remote}
    if spec.arg_params:
        nested = {k: v for k, v in merged.items() if k not in spec.params_flat}
        kwargs.update({k: v for k, v in merged.items() if k in spec.params_flat})
        if nested:
            kwargs[spec.arg_params] = nested
    else:
        kwargs.update(merged)

    return {"entry": spec.entry, "kwargs": kwargs, "pipeline": spec.name}


def requirements(spec: PipelineSpec, pip_extra: list | None = None) -> list:
    """Requirement list installed on the worker: the spec's plus the run's."""
    return list(spec.pip) + list(pip_extra or [])
