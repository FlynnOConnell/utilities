"""Reading the Google account: projects, APIs, buckets and the console links."""

from __future__ import annotations

import pytest

from imgui_cloud import account


class FakeResponse:
    """Enough of a requests response for the account calls to read."""

    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code
        self.reason = "Bad Request"
        self.url = "https://example.invalid"

    def json(self):
        return self.body


class FakeSession:
    """Records the URLs asked for and replays canned bodies in order."""

    def __init__(self):
        self.responses = []
        self.urls = []
        self.body_sent = None

    def reply(self, *responses) -> None:
        """Queue the bodies the next calls should return, in order."""
        self.responses += responses

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.responses.pop(0)

    def post(self, url, **kwargs):
        self.urls.append(url)
        self.body_sent = kwargs.get("json")
        return self.responses.pop(0)


def refuse_api(credentials):
    """Stand-in for a project whose Resource Manager API is off."""
    raise RuntimeError("Cloud Resource Manager API has not been used")


def refuse_api_forbidden(credentials):
    """Stand-in for an account that may not list projects at all."""
    raise RuntimeError("no permission")


@pytest.fixture
def session(monkeypatch):
    """Swap the authorized session for one whose replies the test queues up."""
    fake = FakeSession()
    monkeypatch.setattr(account, "session_for", lambda credentials: fake)
    monkeypatch.setattr(account.time, "sleep", lambda seconds: None)
    return fake


def test_project_label_prefers_the_display_name():
    project = account.Project(
        project_id="pml-subcellular-imaging",
        project_number="1039039069680",
        display_name="PML-subcellular-imaging",
    )
    assert project.label == "PML-subcellular-imaging  (pml-subcellular-imaging)"
    assert account.Project(project_id="p-1").label == "p-1"


def test_projects_from_api_pulls_the_number_out_of_the_resource_name(session):
    session.reply(
        FakeResponse(
            {
                "projects": [
                    {
                        "name": "projects/1039039069680",
                        "projectId": "pml-subcellular-imaging",
                        "displayName": "PML-subcellular-imaging",
                    }
                ]
            }
        )
    )
    found = account.projects_from_api(object())
    assert found[0].project_number == "1039039069680"
    assert found[0].project_id == "pml-subcellular-imaging"


def test_projects_from_api_follows_page_tokens(session):
    session.reply(
        FakeResponse(
            {
                "projects": [{"name": "projects/1", "projectId": "a"}],
                "nextPageToken": "more",
            }
        ),
        FakeResponse({"projects": [{"name": "projects/2", "projectId": "b"}]}),
    )
    assert [p.project_id for p in account.projects_from_api(object())] == ["a", "b"]
    assert len(session.urls) == 2


def test_list_projects_falls_back_to_gcloud_when_the_api_refuses(monkeypatch):
    monkeypatch.setattr(account, "projects_from_api", refuse_api)
    monkeypatch.setattr(account, "path_gcloud", lambda: "gcloud")
    monkeypatch.setattr(
        account,
        "run_gcloud",
        lambda *args, **kwargs: (
            '[{"projectId": "p-1", "projectNumber": "42", "name": "One"}]'
        ),
    )
    assert account.list_projects(object())[0].project_number == "42"


def test_list_projects_reports_the_api_error_when_there_is_no_gcloud(monkeypatch):
    monkeypatch.setattr(account, "projects_from_api", refuse_api_forbidden)
    monkeypatch.setattr(account, "path_gcloud", lambda: "")
    with pytest.raises(RuntimeError, match="no permission"):
        account.list_projects(object())


def test_services_state_reads_one_endpoint_per_api(session):
    session.reply(
        FakeResponse({"state": "ENABLED"}), FakeResponse({"state": "DISABLED"})
    )
    state = account.services_state(object(), "p-1")
    assert state == {"compute.googleapis.com": True, "storage.googleapis.com": False}
    assert all("/projects/p-1/services/" in url for url in session.urls)


def test_failed_calls_raise_the_api_message(session):
    session.reply(FakeResponse({"error": {"message": "billing is not enabled"}}, 403))
    with pytest.raises(RuntimeError, match="billing is not enabled"):
        account.services_state(object(), "p-1")


def test_enable_services_waits_for_the_operation(session):
    session.reply(
        FakeResponse({"name": "operations/abc"}),
        FakeResponse({"name": "operations/abc", "done": True}),
    )
    account.enable_services(object(), "p-1", timeout=5)
    assert session.body_sent == {"serviceIds": list(account.APIS_REQUIRED)}
    assert session.urls[-1].endswith("operations/abc")


def test_enable_services_surfaces_an_operation_error(session):
    session.reply(
        FakeResponse(
            {"name": "operations/abc", "done": True, "error": {"message": "no billing"}}
        )
    )
    with pytest.raises(RuntimeError, match="no billing"):
        account.enable_services(object(), "p-1", timeout=5)


def test_project_id_comes_out_of_a_pasted_console_link():
    url = "https://console.cloud.google.com/storage/browser?project=pml-subcellular-imaging&hl=en"
    assert account.project_id_in(url) == "pml-subcellular-imaging"
    assert account.project_id_in("/v1/projects/pml-subcellular-imaging/services") == (
        "pml-subcellular-imaging"
    )
    assert account.project_id_in("nothing here") == ""


def test_suggested_names_are_derived_from_the_project():
    assert (
        account.bucket_suggested("pml-subcellular-imaging")
        == "pml-subcellular-imaging-imgui-cloud"
    )
    assert account.bucket_suggested("") == ""
    assert (
        account.service_account_default("1039039069680")
        == "1039039069680-compute@developer.gserviceaccount.com"
    )
    assert account.service_account_default("") == ""


def test_run_gcloud_without_the_sdk_says_so(monkeypatch):
    monkeypatch.setattr(account, "path_gcloud", lambda: "")
    with pytest.raises(RuntimeError, match="not installed"):
        account.run_gcloud("projects", "list")


MESSAGE_CONSENT = (
    "ERROR: There was a problem with web authentication. Try running again with "
    "--no-browser.  "
    "ERROR: (gcloud.auth.application-default.login) The "
    "[https://www.googleapis.com/auth/cloud-platform] scope is required but was "
    "not consented. Please run the login command again and consent in the login "
    "page."
)


def test_the_consent_hint_wins_over_the_web_auth_one():
    """gcloud prints both; only the unticked consent box is actionable."""
    hint = account.hint_for(MESSAGE_CONSENT)
    assert "Select all" in hint
    assert account.hint_for("There was a problem with web authentication").startswith(
        "The browser could not hand"
    )


def test_a_disabled_api_points_at_the_button_that_enables_it():
    hint = account.hint_for(
        "Compute Engine API has not been used in project 1039039069680 before"
    )
    assert "Turn both on" in hint


def test_an_unrecognised_failure_gets_no_hint():
    assert account.hint_for("something nobody has seen before") == ""
    assert account.hint_for("") == ""


def test_an_interactive_login_does_not_wait_on_the_browser(monkeypatch):
    spawned = []
    monkeypatch.setattr(account, "path_gcloud", lambda: "gcloud")
    monkeypatch.setattr(
        account.subprocess, "Popen", lambda args, **kwargs: spawned.append(args)
    )
    monkeypatch.setattr(account, "run_gcloud", lambda *args, **kwargs: "")
    account.login_gcloud(interactive=True)
    assert spawned == [["gcloud", "auth", "application-default", "login"]]


QUOTA_INFO_A100_ZONE = {
    "quotaId": "NVIDIA-A100-GPUS-per-project-zone",
    "metric": "compute.googleapis.com/nvidia_a100_gpus",
    "dimensions": ["zone"],
    "dimensionsInfos": [{"dimensions": {}, "details": {"value": "-1"}}],
}
QUOTA_INFO_A100 = {
    "quotaId": "NVIDIA-A100-GPUS-per-project-region",
    "metric": "compute.googleapis.com/nvidia_a100_gpus",
    "quotaDisplayName": "NVIDIA A100 GPUs",
    "dimensions": ["region"],
    "dimensionsInfos": [
        {"dimensions": {"region": "us-central1"}, "details": {"value": "0"}},
        {"dimensions": {"region": "us-west4"}, "details": {"value": "8"}},
    ],
}
QUOTA_INFO_CPUS = {
    "quotaId": "CPUS-per-project-region",
    "metric": "compute.googleapis.com/cpus",
    "dimensions": ["region"],
    "dimensionsInfos": [
        {"dimensions": {"region": "us-central1"}, "details": {"value": "24"}}
    ],
}


def test_quota_infos_keeps_the_gpu_entries_and_their_ids(session):
    session.reply(
        FakeResponse({"quotaInfos": [QUOTA_INFO_A100, QUOTA_INFO_CPUS]}),
    )
    found = account.quota_infos(object(), "pml-subcellular-imaging")
    assert set(found) == {"NVIDIA-A100-GPUS-per-project-region"}
    info = account.info_gating(found, "NVIDIA_A100_GPUS")
    assert info.quota_id == "NVIDIA-A100-GPUS-per-project-region"
    assert info.is_regional
    assert info.limit_in("us-central1") == 0.0
    assert info.regions_with(1) == ["us-west4"]


def test_an_unlisted_region_falls_back_to_the_default_entry():
    info = account.quota_info_of(
        {
            "quotaId": "GPUS-ALL-REGIONS-per-project",
            "metric": "compute.googleapis.com/gpus_all_regions",
            "dimensions": [],
            "dimensionsInfos": [{"dimensions": {}, "details": {"value": "0"}}],
        }
    )
    assert info.is_regional is False
    assert info.limit_in("us-central1") == 0.0


def test_requesting_a_quota_sends_the_id_the_amount_and_the_region(session):
    session.reply(FakeResponse({"quotaConfig": {"stateDetail": "pending"}}))
    info = account.quota_info_of(QUOTA_INFO_A100)
    answer = account.request_quota(
        object(),
        "pml-subcellular-imaging",
        info,
        1,
        region="us-central1",
        email="flynn@example.org",
    )
    assert session.body_sent["quotaId"] == "NVIDIA-A100-GPUS-per-project-region"
    assert session.body_sent["quotaConfig"] == {"preferredValue": "1"}
    assert session.body_sent["dimensions"] == {"region": "us-central1"}
    assert session.body_sent["contactEmail"] == "flynn@example.org"
    assert "requested" in answer


def test_asking_twice_reports_the_request_already_in_flight(session):
    session.reply(FakeResponse({"error": {"message": "already exists"}}, 409))
    info = account.quota_info_of(QUOTA_INFO_A100)
    answer = account.request_quota(object(), "p-1", info, 1, region="us-central1")
    assert "already requested" in answer


def test_a_disabled_api_is_recognised_from_googles_own_wording():
    message = (
        "IAM API has not been used in project 1039039069680 before or it is "
        "disabled. Enable it by visiting https://console.developers.google.com"
        "/apis/api/iam.googleapis.com/overview?project=1039039069680"
    )
    assert account.api_disabled_in(message) == "iam.googleapis.com"
    assert account.api_disabled_in("bucket gs://nope: 404") == ""


def test_every_offered_api_says_what_it_allows():
    for api in account.APIS_ALL:
        assert account.PURPOSE_API[api]


def test_the_region_quota_wins_over_its_unlimited_per_zone_twin(session):
    """Compute publishes both; the per-zone one reads unlimited and gates nothing."""
    session.reply(FakeResponse({"quotaInfos": [QUOTA_INFO_A100_ZONE, QUOTA_INFO_A100]}))
    infos = account.quota_infos(object(), "pml-subcellular-imaging")
    assert len(infos) == 2
    gating = account.info_gating(infos, "NVIDIA_A100_GPUS")
    assert gating.quota_id == "NVIDIA-A100-GPUS-per-project-region"
    assert account.info_gating(infos, "NVIDIA_T4_GPUS") is None
