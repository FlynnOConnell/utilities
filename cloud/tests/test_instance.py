"""The instance resource this package asks Compute Engine to create."""

from __future__ import annotations

import pytest

from imgui_cloud import config as config_module
from imgui_cloud import instance as instance_module

compute_v1 = pytest.importorskip("google.cloud.compute_v1")


def build(machine=None, **kwargs):
    defaults = dict(
        machine=machine or config_module.MachineConfig(),
        zone="us-central1-a",
        name="imgui-cloud-run-1",
        image_self_link="projects/dlvm/global/images/img",
        startup_script="#!/bin/bash\ntrue\n",
    )
    defaults.update(kwargs)
    return instance_module.build_instance(**defaults)


def test_the_box_is_an_a100_with_a_scratch_disk():
    resource = build()
    assert resource.machine_type.endswith("/a2-highgpu-1g")
    assert len(resource.disks) == 2
    boot, data = resource.disks
    assert boot.boot is True and boot.auto_delete is True
    assert data.boot is False
    assert data.device_name == "imgui-cloud-data"
    assert data.initialize_params.disk_size_gb == 1000


def test_a2_types_do_not_declare_guest_accelerators():
    assert not build().guest_accelerators


def test_other_machine_types_do_declare_them():
    machine = config_module.MachineConfig(machine_type="n1-standard-8")
    resource = build(machine=machine)
    assert resource.guest_accelerators[0].accelerator_count == 1
    assert resource.guest_accelerators[0].accelerator_type.endswith("nvidia-tesla-a100")


def test_the_instance_deletes_itself_at_the_runtime_cap():
    machine = config_module.MachineConfig(max_runtime_min=90)
    resource = build(machine=machine)
    assert resource.scheduling.max_run_duration.seconds == 5400
    assert resource.scheduling.instance_termination_action == "DELETE"


def test_spot_is_requested_as_a_provisioning_model():
    assert build().scheduling.provisioning_model == "SPOT"
    on_demand = config_module.MachineConfig(spot=False)
    assert build(machine=on_demand).scheduling.provisioning_model != "SPOT"


def test_scratch_disk_survives_teardown_only_when_asked():
    assert build().disks[1].auto_delete is True
    keep = config_module.MachineConfig(keep_data_disk=True)
    assert build(machine=keep).disks[1].auto_delete is False


def test_an_existing_disk_is_attached_rather_than_created():
    machine = config_module.MachineConfig(data_disk_name="lab-scratch")
    data = build(machine=machine).disks[1]
    assert data.source.endswith("/disks/lab-scratch")
    assert data.auto_delete is False


def test_every_worker_carries_the_managed_by_label():
    resource = build(labels={"run-id": "Run-1", "user": "Flynn.OConnell@x.com"})
    assert resource.labels["managed-by"] == "imgui-cloud"
    assert resource.labels["run-id"] == "run-1"
    assert resource.labels["user"] == "flynn-oconnell-x-com"


def test_the_startup_script_rides_in_instance_metadata():
    resource = build(startup_script="#!/bin/bash\necho hi\n")
    keys = {item.key: item.value for item in resource.metadata.items}
    assert keys["startup-script"].endswith("echo hi\n")
    assert keys["install-nvidia-driver"] == "True"


def test_instance_names_are_dns_safe():
    assert instance_module.instance_name("MK355_run/2") == "imgui-cloud-mk355-run-2"
    assert len(instance_module.instance_name("x" * 100)) <= 63
