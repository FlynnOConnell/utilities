"""Every branch of the account panel, drawn in a headless imgui frame.

A window is not needed to catch the mistakes that matter here: an unbalanced
begin/end, a tree that is never popped, a field drawn before its state exists.
"""

from __future__ import annotations

import pytest

from imgui_cloud import account
from imgui_cloud.gui.login import LoginPanel
from imgui_cloud.gui.setup import SetupState

PROJECT = account.Project(
    project_id="pml-subcellular-imaging",
    project_number="1039039069680",
    display_name="PML-subcellular-imaging",
)


@pytest.fixture
def panel():
    """A panel that never starts a worker thread."""
    made = LoginPanel()
    made._start = lambda task: None
    return made


def test_the_panel_draws_before_anyone_signs_in(panel, render):
    render(panel.draw)
    render(panel._draw_settings)


def test_the_panel_draws_with_a_project_loaded(panel, render):
    panel.state = SetupState(
        gcloud=True,
        email="flynn@example.org",
        projects=[PROJECT],
        apis={api: True for api in account.APIS_REQUIRED},
        buckets=["pml-subcellular-imaging-imgui-cloud"],
        accounts_service=["1039039069680-compute@developer.gserviceaccount.com"],
        zones_by_accelerator={"nvidia-tesla-a100": ["us-central1-a", "us-west1-b"]},
        quotas={"NVIDIA_A100_GPUS": 2.0, "PREEMPTIBLE_NVIDIA_A100_GPUS": 0.0},
    )
    panel.choose_project(PROJECT, read_details=False)
    render(panel.draw)
    render(panel._draw_settings)


def test_every_step_control_draws(panel, render):
    """One frame each: the panel only ever draws the step you are standing on."""
    panel.state = SetupState(gcloud=True, email="flynn@example.org", projects=[PROJECT])
    panel.choose_project(PROJECT, read_details=False)
    for draw in (
        panel._draw_control_install,
        panel._draw_control_signin,
        panel._draw_control_project,
        panel._draw_control_apis,
        panel._draw_control_bucket,
        panel._draw_control_quota,
        panel._draw_control_verify,
    ):
        render(draw)


def test_problems_and_warnings_draw(panel, render):
    panel.error = "gcloud: You do not currently have an active account selected"
    panel.warnings = ["buckets: 403 Cloud Storage API has not been used"]
    panel.status.problems = ["bucket gs://nope: 404"]
    render(panel.draw)


def test_a_console_sign_in_says_it_is_waiting(panel, render):
    panel.waiting_console = True
    render(panel.draw)


def test_a_busy_panel_disables_its_controls(panel, render):
    panel.busy = "signin"
    render(panel.draw)
