"""The account panel's autofill: what picking a project fills in for you."""

from __future__ import annotations

from pathlib import Path

import pytest

from imgui_cloud import account, credentials as credentials_module
from imgui_cloud.gui.login import LoginPanel

PROJECT = account.Project(
    project_id="pml-subcellular-imaging",
    project_number="1039039069680",
    display_name="PML-subcellular-imaging",
)
PROJECT_OTHER = account.Project(project_id="other-project", project_number="7")


@pytest.fixture
def panel():
    """A panel that never starts a worker thread."""
    made = LoginPanel()
    made._start = lambda task: None
    return made


def refuse(*args, **kwargs):
    """Stand-in for an API call the project has not enabled yet."""
    raise RuntimeError("Cloud Storage API has not been used")


def test_choosing_a_project_fills_in_number_name_and_service_account(panel):
    panel.choose_project(PROJECT, read_details=False)
    assert panel.profile.project_id == "pml-subcellular-imaging"
    assert panel.profile.project_number == "1039039069680"
    assert panel.profile.project_name == "PML-subcellular-imaging"
    assert panel.profile.service_account_email == (
        "1039039069680-compute@developer.gserviceaccount.com"
    )
    assert panel.text_bucket_new == "pml-subcellular-imaging-imgui-cloud"


def test_switching_projects_drops_what_belonged_to_the_last_one(panel):
    panel.choose_project(PROJECT, read_details=False)
    panel.profile.bucket = "pml-staging"
    panel.state.buckets = ["pml-staging"]
    panel.state.quotas = {"NVIDIA_A100_GPUS": 4.0}
    panel.choose_project(PROJECT_OTHER, read_details=False)
    assert panel.profile.bucket == ""
    assert panel.state.buckets == []
    assert panel.state.quotas == {}


def test_a_single_visible_project_is_adopted_without_being_asked(panel):
    panel.state.projects = [PROJECT]
    panel._adopt_project_known()
    assert panel.profile.project_id == "pml-subcellular-imaging"


def test_a_stored_project_gains_its_number_from_the_listing(panel):
    panel.profile.project_id = "pml-subcellular-imaging"
    panel.state.projects = [PROJECT_OTHER, PROJECT]
    panel._adopt_project_known()
    assert panel.profile.project_number == "1039039069680"


def test_reading_the_project_fills_apis_buckets_and_quota(panel, monkeypatch):
    panel.choose_project(PROJECT, read_details=False)
    monkeypatch.setattr(
        account,
        "services_state",
        lambda credentials, project, services: {api: True for api in services},
    )
    monkeypatch.setattr(
        account, "list_buckets", lambda credentials, project: ["imgui-cloud-staging"]
    )
    monkeypatch.setattr(
        account,
        "quotas_gpu",
        lambda credentials, project, region: {
            "NVIDIA_A100_GPUS": 0.0,
            "PREEMPTIBLE_NVIDIA_A100_GPUS": 2.0,
            "NVIDIA_L4_GPUS": 8.0,
        },
    )
    monkeypatch.setattr(
        account, "list_service_accounts", lambda credentials, project: ["a@b.iam"]
    )
    monkeypatch.setattr(
        account,
        "quotas_project",
        lambda credentials, project: {"GPUS_ALL_REGIONS": 8.0},
    )
    monkeypatch.setattr(
        account,
        "quota_infos",
        lambda credentials, project: {
            "NVIDIA-A100-GPUS-per-project-region": account.QuotaInfo(
                quota_id="NVIDIA-A100-GPUS-per-project-region",
                metric="NVIDIA_A100_GPUS",
                dimensions=("region",),
                limits={"us-central1": 0.0, "us-west4": 8.0},
            )
        },
    )
    monkeypatch.setattr(
        account,
        "zones_by_accelerator",
        lambda credentials, project: {"nvidia-tesla-a100": ["us-central1-a"]},
    )
    panel._read_details(object())
    assert panel.state.apis_ok
    assert panel.profile.bucket == "imgui-cloud-staging"
    assert panel.state.quota_a100 == 2.0
    assert panel.state.quota_for("NVIDIA_L4_GPUS") == 8.0
    assert panel.state.zones == ["us-central1-a"]
    assert panel.state.gpus_all_regions == 8.0
    assert panel.state.info_for("NVIDIA_A100_GPUS").regions_with(1) == ["us-west4"]
    assert panel.warnings == []


def test_one_failing_call_becomes_a_warning_not_a_crash(panel, monkeypatch):
    panel.choose_project(PROJECT, read_details=False)
    monkeypatch.setattr(
        account,
        "services_state",
        lambda credentials, project, services: {api: True for api in services},
    )
    monkeypatch.setattr(account, "list_buckets", refuse)
    monkeypatch.setattr(account, "quotas_gpu", refuse)
    monkeypatch.setattr(account, "quotas_project", refuse)
    monkeypatch.setattr(account, "quota_infos", refuse)
    monkeypatch.setattr(account, "list_service_accounts", refuse)
    monkeypatch.setattr(account, "zones_by_accelerator", refuse)
    panel._read_details(object())
    assert panel.state.buckets == []
    assert any("buckets" in warning for warning in panel.warnings)


def test_several_buckets_leave_the_choice_open(panel):
    panel.state.buckets = ["one", "two"]
    panel._choose_bucket_default()
    assert panel.profile.bucket == ""


def test_a_task_failure_is_kept_as_text_and_frees_the_panel(panel):
    panel._run(refuse)
    assert "Cloud Storage API" in panel.error
    assert panel.busy == ""


def test_signing_out_forgets_the_in_app_token_only(panel):
    path_token = Path(credentials_module.filepath_token_user())
    path_token.write_text("{}")
    panel.sign_out()
    assert not path_token.exists()
    assert panel.status.ok is False


def test_a_failed_sign_in_is_answered_with_what_to_do_about_it(panel):
    panel.error = (
        "ERROR: (gcloud.auth.application-default.login) The "
        "[https://www.googleapis.com/auth/cloud-platform] scope is required but "
        "was not consented."
    )
    assert "Select all" in panel.hints()[0]


def test_each_hint_is_offered_once(panel):
    panel.error = "scope is required but was not consented"
    panel.status.problems = ["compute us-central1-a: scope is required"]
    assert len(panel.hints()) == 1


def test_a_console_sign_in_is_picked_up_when_the_credentials_land(panel, monkeypatch):
    started = []
    panel._start = lambda task: started.append(task.__name__)
    panel.waiting_console = True
    monkeypatch.setattr(credentials_module, "credentials_available", lambda p: False)
    panel._poll_console_signin()
    assert started == []
    monkeypatch.setattr(credentials_module, "credentials_available", lambda p: True)
    panel._time_polled = 0.0
    panel._poll_console_signin()
    assert started == ["_task_account"]
    assert panel.waiting_console is False
