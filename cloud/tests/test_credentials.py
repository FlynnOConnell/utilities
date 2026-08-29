"""The profile store: what gets written, what deliberately does not."""

from __future__ import annotations

import json

import pytest

from imgui_cloud import credentials as credentials_module


def test_save_and_load_roundtrip():
    profile = credentials_module.CloudProfile(
        name="lab", project_id="mbo-imaging", bucket="mbo-scratch", user_email="a@b.c"
    )
    credentials_module.save_profile(profile)
    loaded = credentials_module.load_profile("lab")
    assert loaded.project_id == "mbo-imaging"
    assert loaded.bucket == "mbo-scratch"
    assert credentials_module.active_profile_name() == "lab"


def test_key_contents_are_never_copied_into_the_store(tmp_path):
    filepath_key = tmp_path / "key.json"
    filepath_key.write_text(json.dumps({"private_key": "SECRET", "project_id": "p"}))
    profile = credentials_module.CloudProfile(
        name="lab", filepath_service_account_key=str(filepath_key)
    )
    credentials_module.save_profile(profile)
    written = open(credentials_module.filepath_profiles()).read()
    assert "SECRET" not in written
    assert str(filepath_key) in written.replace("\\\\", "\\")


def test_missing_profile_yields_defaults_rather_than_raising():
    profile = credentials_module.load_profile("never-created")
    assert profile.name == "never-created"
    assert profile.zone == "us-central1-a"


def test_delete_profile_moves_the_active_pointer():
    credentials_module.save_profile(credentials_module.CloudProfile(name="a"))
    credentials_module.save_profile(credentials_module.CloudProfile(name="b"))
    assert credentials_module.active_profile_name() == "b"
    credentials_module.delete_profile("b")
    assert credentials_module.active_profile_name() == "a"


def test_unknown_keys_in_the_store_are_ignored():
    credentials_module.save_profile(credentials_module.CloudProfile(name="a"))
    path = credentials_module.filepath_profiles()
    raw = json.loads(open(path).read())
    raw["profiles"]["a"]["from_a_newer_version"] = 1
    open(path, "w").write(json.dumps(raw))
    assert credentials_module.load_profile("a").name == "a"


def test_missing_key_file_fails_loudly_instead_of_falling_back_to_adc():
    profile = credentials_module.CloudProfile(
        filepath_service_account_key="/no/such/key.json"
    )
    with pytest.raises(FileNotFoundError):
        credentials_module.credentials_for(profile)


def test_check_profile_collects_every_problem():
    status = credentials_module.check_profile(
        credentials_module.CloudProfile(
            filepath_service_account_key="/no/such/key.json"
        )
    )
    assert status.ok is False
    assert any("project id" in p for p in status.problems)
    assert any("bucket" in p for p in status.problems)
    assert "not signed in" or status.summary()


def test_uri_run_and_region():
    profile = credentials_module.CloudProfile(bucket="b", prefix="p", zone="us-west1-b")
    assert profile.uri_run("r1") == "gs://b/p/r1"
    assert profile.region == "us-west1"


class FakeInstancesClient:
    """Stands in for compute_v1.InstancesClient, with its real kwarg rules."""

    requests: list = []

    def __init__(self, credentials=None):
        pass

    def list(self, request=None, **kwargs):
        """The generated client flattens only project and zone; the rest is a request."""
        if kwargs:
            raise TypeError(f"Unexpected keyword {sorted(kwargs)}")
        FakeInstancesClient.requests.append(request)
        return []


def test_the_compute_probe_asks_through_a_request_not_a_stray_keyword(monkeypatch):
    """max_results is a request field: passing it as a kwarg is a TypeError."""
    from google.cloud import compute_v1

    FakeInstancesClient.requests = []
    monkeypatch.setattr(compute_v1, "InstancesClient", FakeInstancesClient)
    monkeypatch.setattr(
        credentials_module, "credentials_for", lambda profile: (object(), "p-1")
    )
    profile = credentials_module.CloudProfile(project_id="p-1", zone="us-central1-a")
    status = credentials_module.check_profile(profile)

    assert status.compute_ok is True
    assert not [p for p in status.problems if "compute" in p]
    asked = FakeInstancesClient.requests[0]
    assert (asked.project, asked.zone, asked.max_results) == (
        "p-1",
        "us-central1-a",
        1,
    )
