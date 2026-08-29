"""The worker startup script - the part that has to be right before it runs."""

from __future__ import annotations

from imgui_cloud import startup


def script(**kwargs) -> str:
    defaults = dict(uri_run="gs://bkt/pre/run-1", run_id="run-1")
    defaults.update(kwargs)
    return startup.build_startup_script(**defaults)


def test_no_placeholders_survive_rendering():
    text = script(requirements=["mbo_utilities"], env={"FOO": "bar"})
    assert "__" not in text.replace("__pycache__", "")


def test_every_stage_of_the_box_lifecycle_is_present():
    text = script(requirements=["mbo_utilities"])
    for fragment in (
        "mkfs.ext4",
        "mount -o discard,defaults",
        'gcloud storage rsync --recursive "$GS_RUN/input"',
        'gcloud storage rsync --recursive "$DATA/output"',
        "worker_main.py",
        "umount",
        "state DONE",
        "state FAILED",
    ):
        assert fragment in text, fragment


def test_the_scratch_disk_path_matches_the_attached_device_name():
    text = script()
    assert f"/dev/disk/by-id/google-{startup.DEVICE_NAME_DATA}" in text


def test_self_delete_is_on_by_default_and_can_be_turned_off():
    assert "SELF_DELETE=1" in script()
    assert "SELF_DELETE=0" in script(self_delete=False)
    assert "gcloud --quiet compute instances delete" in script()


def test_requirements_and_env_are_shell_quoted():
    text = script(
        requirements=["masknmf[multisession] @ git+https://x/y.git@branch"],
        env={"TOKEN": "a b; rm -rf /"},
    )
    assert "'masknmf[multisession] @ git+https://x/y.git@branch'" in text
    assert "export TOKEN='a b; rm -rf /'" in text


def test_no_requirements_still_produces_a_valid_block():
    text = script(requirements=[])
    assert "no extra requirements" in text
    assert "pip install \n" not in text


def test_pip_failure_aborts_the_run():
    text = script(requirements=["mbo_utilities"])
    assert 'fail "pip install failed"' in text


def test_the_worker_builds_its_environment_with_uv_not_the_image_python():
    """The image ships 3.10; the lab packages need 3.12, so uv fetches one."""
    script = startup.build_startup_script(
        "gs://b/p/r", "r", requirements=["git+https://example.invalid/x.git@main"]
    )
    assert "uv venv --python 3.12" in script
    assert "--torch-backend auto" in script
    assert "git+https://example.invalid/x.git@main" in script


def test_a_pipeline_environment_is_the_same_object_everywhere():
    from imgui_cloud import environment

    spec = environment.spec_for("suite2p", python="3.12", dir_venv="/opt/x")
    assert spec.filepath_python.endswith("python") or spec.filepath_python.endswith(
        "python.exe"
    )
    standalone = environment.script_bootstrap(spec, standalone=True)
    assert standalone.startswith("#!/usr/bin/env bash")
    assert "fail()" in standalone
    assert all(requirement in standalone for requirement in spec.requirements)
