"""The first-run checklist: what it says is done, and what it hands you."""

from __future__ import annotations

from imgui_cloud.credentials import CloudProfile, ProfileStatus
from imgui_cloud.gui.setup import (
    APIS_REQUIRED,
    SetupState,
    index_current,
    steps_for,
    url_console,
)

PROFILE_PML = CloudProfile(
    project_id="pml-subcellular-imaging",
    project_number="1039039069680",
    project_name="PML-subcellular-imaging",
    bucket="pml-staging",
)


def titles(steps) -> list:
    return [step.title for step in steps]


def test_a_machine_with_the_sdk_starts_at_the_sign_in():
    steps = steps_for(CloudProfile(), ProfileStatus(), SetupState(gcloud=True))
    assert not any(step.done for step in steps)
    assert steps[index_current(steps)].title == "Sign in with Google"


def test_a_machine_without_the_sdk_is_told_to_install_it_first():
    steps = steps_for(CloudProfile(), ProfileStatus(), SetupState(gcloud=False))
    assert titles(steps)[0] == "Install the Google Cloud SDK"
    assert steps[0].command and steps[0].url


def test_the_install_step_disappears_once_a_key_file_signs_in():
    state = SetupState(gcloud=False, email="flynn@example.org")
    steps = steps_for(PROFILE_PML, ProfileStatus(credentials_ok=True), state)
    assert "Install the Google Cloud SDK" not in titles(steps)


def test_probe_results_tick_the_matching_steps():
    status = ProfileStatus(credentials_ok=True, compute_ok=True)
    done = {
        s.title: s.done for s in steps_for(PROFILE_PML, status, SetupState(gcloud=True))
    }
    assert done["Sign in with Google"]
    assert done["Pick your project"]
    assert done["Turn on Compute Engine and Cloud Storage"]
    assert not done["Choose the staging bucket"]


def test_listed_state_beats_the_probe_for_apis_and_buckets():
    state = SetupState(
        gcloud=True,
        email="flynn@example.org",
        apis={api: True for api in APIS_REQUIRED},
        buckets=["pml-staging"],
        quotas={"NVIDIA_A100_GPUS": 2.0},
    )
    steps = steps_for(PROFILE_PML, ProfileStatus(credentials_ok=True), state)
    undone = [s.title for s in steps if not s.done]
    assert undone == ["Check the connection"]


def test_quota_is_a_real_check_once_the_region_has_been_read():
    state = SetupState(gcloud=True, quotas={"NVIDIA_A100_GPUS": 0.0})
    done = {s.title: s.done for s in steps_for(PROFILE_PML, ProfileStatus(), state)}
    assert not done["Request A100 quota"]
    state.quotas = {"NVIDIA_A100_GPUS": 0.0, "PREEMPTIBLE_NVIDIA_A100_GPUS": 4.0}
    done = {s.title: s.done for s in steps_for(PROFILE_PML, ProfileStatus(), state)}
    assert done["Request A100 quota"]


def test_an_unread_region_leaves_the_quota_unknown_rather_than_zero():
    assert SetupState().quota_a100 == -1.0
    assert SetupState(quotas={"NVIDIA_L4_GPUS": 8.0}).quota_a100 == -1.0


def test_the_project_step_shows_the_name_id_and_number():
    steps = steps_for(PROFILE_PML, ProfileStatus(), SetupState(gcloud=True))
    note = next(s.note for s in steps if s.title == "Pick your project")
    assert "PML-subcellular-imaging" in note
    assert "pml-subcellular-imaging" in note
    assert "1039039069680" in note


def test_commands_carry_the_project_and_region():
    profile = CloudProfile(
        project_id="pml-subcellular-imaging", bucket="pml-staging", zone="us-east1-b"
    )
    commands = {s.title: s.command for s in steps_for(profile, ProfileStatus())}
    enable = commands["Turn on Compute Engine and Cloud Storage"]
    assert all(api in enable for api in APIS_REQUIRED)
    assert "--project=pml-subcellular-imaging" in enable
    assert (
        commands["Choose the staging bucket"]
        == "gcloud storage buckets create gs://pml-staging"
        " --project=pml-subcellular-imaging --location=us-east1"
    )


def test_a_project_without_an_id_still_yields_usable_commands():
    """Copying a command before the id is known must not emit `--project=`."""
    commands = [s.command for s in steps_for(CloudProfile(), ProfileStatus())]
    assert not any("--project=" in c and c.endswith("--project=") for c in commands)
    assert "BUCKET-NAME" in " ".join(commands)


def test_console_urls_are_scoped_to_the_project_when_there_is_one():
    assert url_console("/storage/browser", "pml-subcellular-imaging") == (
        "https://console.cloud.google.com/storage/browser"
        "?project=pml-subcellular-imaging"
    )
    assert url_console("/storage/browser") == (
        "https://console.cloud.google.com/storage/browser"
    )
