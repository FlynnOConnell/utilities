"""
imgui_cloud - ephemeral GPU workers on Google Cloud, from a panel or a shell.

One run is: upload a folder to a bucket, create an A100 box with a scratch disk
attached, let it pull the data, run a registered pipeline, push the results
back, and delete itself. Nothing stays up, and nothing needs an SSH session.

From code::

    from imgui_cloud import CloudConfig, CloudRun, load_profile

    config = CloudConfig()
    config.io.input = "/data/raw"
    config.io.output = "/data/results"
    config.job.pipeline = "masknmf"

    run = CloudRun(config, profile=load_profile())
    run.start()
    run.wait()

From a GUI, embed the panel in an existing imgui app::

    from imgui_cloud.gui import CloudPanel

    panel = CloudPanel()          # in setup
    panel.draw()                  # once per frame, inside a window or tab

From a shell::

    imgui-cloud login
    imgui-cloud init /data/raw
    imgui-cloud run cloud.toml
"""

from __future__ import annotations

from imgui_cloud.config import (
    CloudConfig,
    IOConfig,
    JobConfig,
    MachineConfig,
    default_config,
    from_toml,
    write_template,
)
from imgui_cloud.credentials import (
    CloudProfile,
    ProfileStatus,
    check_profile,
    load_profile,
    load_profiles,
    save_profile,
)
from imgui_cloud.pipelines import (
    PipelineSpec,
    available as available_pipelines,
    register,
)
from imgui_cloud.run import CloudRun, RunState, execute, teardown_orphans

__version__ = "0.1.0"

__all__ = [
    "CloudConfig",
    "CloudProfile",
    "CloudRun",
    "IOConfig",
    "JobConfig",
    "MachineConfig",
    "PipelineSpec",
    "ProfileStatus",
    "RunState",
    "available_pipelines",
    "check_profile",
    "default_config",
    "execute",
    "from_toml",
    "load_profile",
    "load_profiles",
    "register",
    "save_profile",
    "teardown_orphans",
    "write_template",
]
