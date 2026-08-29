"""
Compute Engine lifecycle: bring up an A100 box, watch it, take it back down.

One instance per run. It boots from a Deep Learning VM image (CUDA and gcloud
already installed), carries a second disk that the startup script formats and
mounts as scratch, and runs the whole job from its metadata startup script.

Three separate mechanisms guarantee it goes away:

1. the startup script deletes the instance when the job ends,
2. :func:`delete_worker` deletes it when the client finishes or is cancelled,
3. ``scheduling.max_run_duration`` with ``instance_termination_action=DELETE``
   deletes it at ``max_runtime_min`` no matter what either of those did.

Only labelled instances (``managed-by=imgui-cloud``) are ever listed or deleted
here, so this can never touch a VM someone else created.
"""

from __future__ import annotations


import re

LABEL_MANAGED = "imgui-cloud"
STATES_ALIVE = ("PROVISIONING", "STAGING", "RUNNING", "REPAIRING")


def sanitize_label(value: str) -> str:
    """Coerce a string into something Compute Engine accepts as a label value."""
    cleaned = re.sub(r"[^a-z0-9_-]", "-", str(value).lower()).strip("-")
    return cleaned[:63]


def instance_name(run_id: str) -> str:
    """
    Instance name for a run id: RFC1035, so no underscores and 63 chars at most.

    Label values tolerate underscores; instance names do not, which is why this
    is not just :func:`sanitize_label`.
    """
    cleaned = re.sub(r"[^a-z0-9-]", "-", f"imgui-cloud-{run_id}".lower())
    return re.sub(r"-+", "-", cleaned)[:63].rstrip("-")


def clients(credentials):
    """``(InstancesClient, ImagesClient)`` bound to ``credentials``."""
    from google.cloud import compute_v1

    return (
        compute_v1.InstancesClient(credentials=credentials),
        compute_v1.ImagesClient(credentials=credentials),
    )


def resolve_image(credentials, image_project: str, image_family: str) -> str:
    """
    Self-link of the newest non-deprecated image in a family.

    Raises
    ------
    RuntimeError
        If the family does not resolve - usually a renamed Deep Learning VM
        family, which is worth failing loudly on rather than booting something
        arbitrary.
    """
    _, images = clients(credentials)
    try:
        image = images.get_from_family(project=image_project, family=image_family)
    except Exception as e:
        raise RuntimeError(
            f"no image family {image_family!r} in project {image_project!r}: {e}"
        ) from e
    return image.self_link


def build_instance(
    machine,
    zone: str,
    name: str,
    image_self_link: str,
    startup_script: str,
    service_account_email: str = "",
    labels: dict | None = None,
    device_name: str = "imgui-cloud-data",
):
    """
    Build the :class:`google.cloud.compute_v1.Instance` resource for a worker.

    Pure construction, no API calls, so the exact shape of a request can be
    asserted in tests without a project.

    Parameters
    ----------
    machine : imgui_cloud.config.MachineConfig
        Hardware and lifetime settings.
    zone : str
        Zone the instance is created in; disk types are zonal URLs.
    name : str
        Instance name.
    image_self_link : str
        Boot image, from :func:`resolve_image`.
    startup_script : str
        Bash handed to the instance as ``startup-script`` metadata.
    service_account_email : str
        Service account attached to the VM; empty uses the project default.
    labels : dict, optional
        Extra labels merged over the managed-by label.
    device_name : str
        Device name of the scratch disk, which fixes its ``/dev/disk/by-id`` path.
    """
    from google.cloud import compute_v1

    disk_boot = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image=image_self_link,
            disk_size_gb=machine.boot_disk_gb,
            disk_type=f"zones/{zone}/diskTypes/{machine.boot_disk_type}",
        ),
    )
    if machine.data_disk_name:
        disk_data = compute_v1.AttachedDisk(
            boot=False,
            auto_delete=False,
            device_name=device_name,
            source=f"zones/{zone}/disks/{machine.data_disk_name}",
        )
    else:
        disk_data = compute_v1.AttachedDisk(
            boot=False,
            auto_delete=not machine.keep_data_disk,
            device_name=device_name,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                disk_size_gb=machine.data_disk_gb,
                disk_type=f"zones/{zone}/diskTypes/{machine.data_disk_type}",
            ),
        )

    scheduling = compute_v1.Scheduling(
        on_host_maintenance="TERMINATE",
        automatic_restart=False,
        max_run_duration=compute_v1.Duration(seconds=machine.max_runtime_min * 60),
        instance_termination_action="DELETE",
    )
    if machine.spot:
        scheduling.provisioning_model = "SPOT"

    instance = compute_v1.Instance(
        name=name,
        machine_type=f"zones/{zone}/machineTypes/{machine.machine_type}",
        disks=[disk_boot, disk_data],
        scheduling=scheduling,
        labels={
            "managed-by": LABEL_MANAGED,
            **{k: sanitize_label(v) for k, v in (labels or {}).items()},
        },
        network_interfaces=[
            compute_v1.NetworkInterface(
                name="global/networks/default",
                access_configs=[
                    compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
                ],
            )
        ],
        service_accounts=[
            compute_v1.ServiceAccount(
                email=service_account_email or "default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        ],
        metadata=compute_v1.Metadata(
            items=[
                compute_v1.Items(key="startup-script", value=startup_script),
                compute_v1.Items(key="install-nvidia-driver", value="True"),
            ]
        ),
    )
    if machine.needs_guest_accelerator:
        instance.guest_accelerators = [
            compute_v1.AcceleratorConfig(
                accelerator_type=f"zones/{zone}/acceleratorTypes/{machine.accelerator_type}",
                accelerator_count=machine.accelerator_count,
            )
        ]
    return instance


def create_worker(
    credentials,
    project: str,
    zone: str,
    instance_resource,
    timeout_s: int = 600,
) -> str:
    """
    Create the instance and block until the API call completes.

    Returns
    -------
    str
        The instance name.

    Raises
    ------
    RuntimeError
        If the create operation reports an error (quota, no A100 capacity in
        the zone, bad machine type). The message is passed through verbatim -
        it is nearly always actionable.
    """
    instances, _ = clients(credentials)
    operation = instances.insert(
        project=project, zone=zone, instance_resource=instance_resource
    )
    operation.result(timeout=timeout_s)
    if getattr(operation, "error_code", None):
        raise RuntimeError(
            f"instance create failed: {operation.error_code} {operation.error_message}"
        )
    return instance_resource.name


def get_worker(credentials, project: str, zone: str, name: str):
    """The instance, or None when it no longer exists."""
    from google.api_core import exceptions

    instances, _ = clients(credentials)
    try:
        return instances.get(project=project, zone=zone, instance=name)
    except exceptions.NotFound:
        return None


def worker_status(credentials, project: str, zone: str, name: str) -> str:
    """Instance status string, or ``"DELETED"`` when it is gone."""
    instance = get_worker(credentials, project, zone, name)
    return instance.status if instance is not None else "DELETED"


def delete_worker(
    credentials, project: str, zone: str, name: str, wait: bool = False
) -> bool:
    """
    Delete the instance. Returns False when it was already gone.

    Idempotent on purpose: teardown runs from several places, and a race with
    the worker's own self-delete must not raise.
    """
    from google.api_core import exceptions

    instances, _ = clients(credentials)
    try:
        operation = instances.delete(project=project, zone=zone, instance=name)
    except exceptions.NotFound:
        return False
    if wait:
        operation.result(timeout=600)
    return True


def list_workers(credentials, project: str, zone: str) -> list:
    """Every instance in the zone that this package created."""
    instances, _ = clients(credentials)
    return [
        instance
        for instance in instances.list(project=project, zone=zone)
        if (instance.labels or {}).get("managed-by") == LABEL_MANAGED
    ]


def serial_output(credentials, project: str, zone: str, name: str) -> str:
    """
    The instance's serial console text.

    Useful when a worker dies before it can push its log to GCS - the startup
    script's early failures land here and nowhere else.
    """
    from google.api_core import exceptions
    from google.cloud import compute_v1

    client = compute_v1.InstancesClient(credentials=credentials)
    try:
        return client.get_serial_port_output(
            project=project, zone=zone, instance=name
        ).contents
    except exceptions.NotFound:
        return ""


def zones_with_a100(
    credentials, project: str, accelerator_type: str = "nvidia-tesla-a100"
) -> list:
    """Zones in the project's region list that advertise the given accelerator."""
    from google.cloud import compute_v1

    client = compute_v1.AcceleratorTypesClient(credentials=credentials)
    found = []
    for zone, scoped in client.aggregated_list(project=project):
        for item in scoped.accelerator_types or []:
            if item.name == accelerator_type:
                found.append(zone.rsplit("/", 1)[-1])
                break
    return sorted(found)
