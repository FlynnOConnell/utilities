"""Config loading, validation, overrides and the cost estimate."""

from __future__ import annotations

import pytest

from imgui_cloud import config as config_module


def test_template_roundtrips_through_the_loader(tmp_path):
    filepath = tmp_path / "cloud.toml"
    dir_input = tmp_path / "raw"
    dir_input.mkdir()
    config_module.write_template(
        str(filepath),
        dir_input=str(dir_input),
        dir_output=str(tmp_path / "out"),
        name="mk355",
    )
    config = config_module.from_toml(str(filepath))
    assert config.io.name == "mk355"
    assert config.io.input == str(dir_input).replace("\\", "/")
    assert config.machine.machine_type == "a2-highgpu-1g"
    assert config.job.pipeline == "masknmf"
    assert config.validate() == []


def test_write_template_refuses_to_clobber(tmp_path):
    filepath = tmp_path / "cloud.toml"
    config_module.write_template(str(filepath))
    with pytest.raises(FileExistsError):
        config_module.write_template(str(filepath))
    config_module.write_template(str(filepath), overwrite=True)


def test_unknown_toml_key_fails_loudly(tmp_path):
    filepath = tmp_path / "cloud.toml"
    filepath.write_text('[machine]\nmachine_typo = "a2-highgpu-1g"\n')
    with pytest.raises(ValueError, match="machine_typo"):
        config_module.from_toml(str(filepath))


def test_validate_reports_every_problem(tmp_path):
    config = config_module.CloudConfig()
    config.io.input = str(tmp_path / "missing")
    config.job.pipeline = "nope"
    config.machine.max_runtime_min = 1
    problems = " ".join(config.validate())
    assert "input does not exist" in problems
    assert "output is empty" in problems
    assert "max_runtime_min" in problems
    assert "unknown pipeline" in problems


def test_merge_overrides_skips_none_and_rejects_typos():
    config = config_module.CloudConfig()
    config_module.merge_overrides(config, {"io.name": "x", "machine.spot": None})
    assert config.io.name == "x"
    assert config.machine.spot is True
    with pytest.raises(KeyError):
        config_module.merge_overrides(config, {"machine.nonesuch": 1})


def test_resolve_output_dir_never_overwrites(tmp_path):
    config = config_module.default_config(dir_output=str(tmp_path), name="run")
    first = config_module.resolve_output_dir(config)
    first.mkdir(parents=True)
    second = config_module.resolve_output_dir(config)
    assert second != first
    assert second.name.endswith("_2")


def test_a100_machines_do_not_ask_for_guest_accelerators():
    machine = config_module.MachineConfig()
    assert machine.needs_guest_accelerator is False
    machine.machine_type = "n1-standard-8"
    assert machine.needs_guest_accelerator is True


def test_spot_is_cheaper_than_on_demand():
    machine = config_module.MachineConfig(spot=True)
    on_demand = config_module.MachineConfig(spot=False)
    assert machine.cost_per_hour_estimate() < on_demand.cost_per_hour_estimate()
    four_gpus = config_module.MachineConfig(machine_type="a2-highgpu-4g", spot=False)
    assert four_gpus.cost_per_hour_estimate() > on_demand.cost_per_hour_estimate() * 3
