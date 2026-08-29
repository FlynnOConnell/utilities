"""The Run tab: choosing a GPU, reading its quota, and getting to Launch."""

from __future__ import annotations

from pathlib import Path

import pytest

from imgui_cloud import account
from imgui_cloud import config as config_module
from imgui_cloud.gui.panel import CloudPanel
from imgui_cloud.gui.setup import SetupState

INFO_A100 = account.QuotaInfo(
    quota_id="NVIDIA-A100-GPUS-per-project-region",
    metric="NVIDIA_A100_GPUS",
    dimensions=("region",),
    limits={"us-central1": 0.0, "us-west4": 8.0},
)
INFO_ALL_REGIONS = account.QuotaInfo(
    quota_id="GPUS-ALL-REGIONS-per-project",
    metric="GPUS_ALL_REGIONS",
    limits={"": 0.0},
)


def state_no_quota() -> SetupState:
    """A signed-in project whose GPU quota is zero everywhere, as new ones are."""
    return SetupState(
        apis={api: True for api in account.APIS_ALL},
        quotas={"NVIDIA_A100_GPUS": 0.0, "PREEMPTIBLE_NVIDIA_A100_GPUS": 0.0},
        quotas_project={"GPUS_ALL_REGIONS": 0.0},
        quota_infos={
            "NVIDIA_A100_GPUS": INFO_A100,
            "GPUS_ALL_REGIONS": INFO_ALL_REGIONS,
        },
        zones_by_accelerator={"nvidia-tesla-a100": ["us-central1-a", "us-west4-b"]},
    )


@pytest.fixture
def panel():
    """A cloud panel whose account never reaches the network."""
    made = CloudPanel(dir_input="")
    made.login._start = lambda task: None
    made.login._started = True
    return made


def test_every_catalog_entry_names_its_quota_metric_after_its_gpu():
    for option in config_module.GPUS:
        family = option.gpu.split()[0].upper()
        assert family in option.metric_quota, option
        assert option.count >= 1
        assert option.usd_per_hour > 0


def test_implicit_gpu_families_do_not_ask_for_a_guest_accelerator():
    for option in config_module.GPUS:
        machine = config_module.MachineConfig(
            machine_type=option.machine_type,
            accelerator_type=option.accelerator_type,
            accelerator_count=option.count,
        )
        expected = not option.machine_type.startswith(("a2-", "a3-", "a4-", "g2-"))
        assert machine.needs_guest_accelerator is expected, option


def test_one_machine_type_can_carry_two_different_gpus():
    t4 = config_module.gpu_option("n1-standard-8", "nvidia-tesla-t4")
    v100 = config_module.gpu_option("n1-standard-8", "nvidia-tesla-v100")
    assert t4.gpu.startswith("T4")
    assert v100.gpu.startswith("V100")
    assert config_module.gpu_option("nothing-like-this") is None


A100 = config_module.gpu_option("a2-highgpu-1g")


def test_a_zero_preemptible_quota_does_not_mean_a_spot_run_is_blocked():
    """Preemptible quota is opt-in; without it spot VMs use the ordinary quota."""
    quotas = {"NVIDIA_A100_GPUS": 8.0, "PREEMPTIBLE_NVIDIA_A100_GPUS": 0.0}
    assert config_module.quota_effective(quotas, A100, spot=True) == (
        8.0,
        "NVIDIA_A100_GPUS",
    )
    assert config_module.quota_effective(quotas, A100, spot=False) == (
        8.0,
        "NVIDIA_A100_GPUS",
    )


def test_a_granted_preemptible_quota_is_the_one_a_spot_run_uses():
    quotas = {"NVIDIA_A100_GPUS": 1.0, "PREEMPTIBLE_NVIDIA_A100_GPUS": 16.0}
    assert config_module.quota_effective(quotas, A100, spot=True) == (
        16.0,
        "PREEMPTIBLE_NVIDIA_A100_GPUS",
    )


def test_an_unread_region_reads_as_unknown_not_as_zero():
    limit, metric = config_module.quota_effective({}, A100, spot=True)
    assert limit == -1.0
    assert metric == "NVIDIA_A100_GPUS"


def test_the_recommendation_is_the_cheapest_box_with_real_quota():
    quotas = {"NVIDIA_A100_GPUS": 4.0, "NVIDIA_L4_GPUS": 0.0, "NVIDIA_T4_GPUS": 0.0}
    best = config_module.recommend_gpu(quotas, {}, "us-central1-a", spot=True)
    assert best.machine_type == "a2-highgpu-1g"


def test_the_recommendation_prefers_a_gpu_the_current_zone_carries():
    quotas = {"NVIDIA_A100_GPUS": 4.0, "NVIDIA_T4_GPUS": 4.0}
    zones = {
        "nvidia-tesla-t4": ["europe-west4-a"],
        "nvidia-tesla-a100": ["us-central1-a"],
    }
    best = config_module.recommend_gpu(quotas, zones, "us-central1-a", spot=True)
    assert best.accelerator_type == "nvidia-tesla-a100"


def test_nothing_is_recommended_when_the_project_has_no_gpu_quota_at_all():
    assert config_module.recommend_gpu({"NVIDIA_A100_GPUS": 0.0}, {}, "z") is None


def test_the_estimate_follows_the_chosen_gpu():
    l4 = config_module.MachineConfig(
        machine_type="g2-standard-8", accelerator_type="nvidia-l4", spot=False
    )
    a100 = config_module.MachineConfig(spot=False)
    h100 = config_module.MachineConfig(
        machine_type="a3-highgpu-8g", accelerator_type="nvidia-h100-80gb", spot=False
    )
    assert l4.cost_per_hour_estimate() < a100.cost_per_hour_estimate()
    assert h100.cost_per_hour_estimate() > a100.cost_per_hour_estimate() * 5


def test_choosing_a_gpu_sets_the_machine_the_accelerator_and_the_count(panel):
    panel.choose_gpu(config_module.gpu_option("a2-ultragpu-2g"))
    machine = panel.config.machine
    assert machine.machine_type == "a2-ultragpu-2g"
    assert machine.accelerator_type == "nvidia-a100-80gb"
    assert machine.accelerator_count == 2
    assert machine.needs_guest_accelerator is False


def test_the_machine_tab_draws_with_no_quota_read_yet(panel, context, render):
    render(panel._draw_machine)


def test_the_machine_tab_draws_the_quota_it_did_read(panel, context, render):
    panel.login.state = SetupState(
        quotas={
            "NVIDIA_A100_GPUS": 0.0,
            "PREEMPTIBLE_NVIDIA_A100_GPUS": 0.0,
            "NVIDIA_L4_GPUS": 8.0,
        },
        zones_by_accelerator={"nvidia-l4": ["us-west1-b", "europe-west4-a"]},
    )
    panel.choose_gpu(config_module.gpu_option("g2-standard-8"))
    render(panel._draw_machine)


def test_the_machine_tab_draws_a_machine_the_catalog_does_not_carry(
    panel, context, render
):
    panel.config.machine.machine_type = "a2-megagpu-16g"
    render(panel._draw_machine)


def test_the_run_tab_draws_before_anything_is_known(panel, context, render):
    render(panel._draw_run)


def test_the_run_tab_offers_a_way_out_when_the_chosen_gpu_has_no_quota(
    panel, context, render
):
    panel.login.state = SetupState(
        quotas={"NVIDIA_A100_GPUS": 0.0, "NVIDIA_T4_GPUS": 4.0},
        zones_by_accelerator={"nvidia-tesla-t4": ["us-central1-a"]},
        quotas_project={"GPUS_ALL_REGIONS": 0.0},
    )
    panel.config.io.input = str(Path.cwd())
    render(panel._draw_run, panel._draw_machine)


def test_using_the_recommendation_switches_the_gpu_and_the_zone(panel):
    panel.login.state = SetupState(
        quotas={"NVIDIA_A100_GPUS": 0.0, "NVIDIA_L4_GPUS": 4.0},
        zones_by_accelerator={"nvidia-l4": ["europe-west4-a"]},
    )
    assert panel.use_recommended_gpu() is True
    assert panel.config.machine.accelerator_type == "nvidia-l4"
    assert panel.login.profile.zone == "europe-west4-a"


def test_the_recommendation_reports_when_there_is_nothing_to_switch_to(panel):
    panel.login.state = SetupState(quotas={"NVIDIA_A100_GPUS": 0.0})
    assert panel.use_recommended_gpu() is False


def test_taking_an_offered_zone_moves_the_profile_and_saves_it(panel):
    from imgui_cloud import credentials as credentials_module

    panel.login.state = SetupState(zones_by_accelerator={"nvidia-l4": ["us-west1-b"]})
    panel.choose_gpu(config_module.gpu_option("g2-standard-8"))
    assert panel.login.profile.zone not in panel.login.state.zones_for("nvidia-l4")
    panel.use_zone("us-west1-b")
    assert panel.login.profile.zone == "us-west1-b"
    assert credentials_module.load_profile(panel.login.profile.name).zone == (
        "us-west1-b"
    )


def test_the_run_form_names_what_the_account_still_owes(panel):
    panel.login.profile.project_id = ""
    panel.login.profile.bucket = ""
    assert panel.problems_profile() == [
        "no project picked yet",
        "no staging bucket yet",
    ]


def test_a_complete_account_falls_back_to_what_the_probe_found(panel):
    panel.login.profile.project_id = "pml-subcellular-imaging"
    panel.login.profile.bucket = "pml-staging"
    panel.login.status.problems = ["bucket gs://pml-staging: 404"]
    assert panel.problems_profile() == ["bucket gs://pml-staging: 404"]


def test_signing_in_opens_the_tabs_even_before_everything_checks_out(panel):
    panel.login.status.credentials_ok = True
    assert panel.login.signed_in is True
    assert panel.login.verified is False
    panel.login.status.ok = True
    assert panel.login.verified is True


def test_the_whole_panel_draws_for_a_half_finished_account(panel, context, render):
    panel.login.status.credentials_ok = True
    panel.login.profile.bucket = ""
    render(panel.draw)


def test_a_project_with_no_quota_asks_for_the_cap_and_the_gpu(panel):
    panel.login.state = state_no_quota()
    panel.choose_gpu(config_module.gpu_option("a2-highgpu-1g"))
    asks = panel.quota_asks(config_module.gpu_option("a2-highgpu-1g"))
    assert [(info.metric, value, region) for info, value, region in asks] == [
        ("GPUS_ALL_REGIONS", 1.0, ""),
        ("NVIDIA_A100_GPUS", 1.0, "us-central1"),
    ]


def test_nothing_is_asked_for_once_the_quota_covers_the_run(panel):
    state = state_no_quota()
    state.quotas["NVIDIA_A100_GPUS"] = 4.0
    state.quotas_project["GPUS_ALL_REGIONS"] = 4.0
    panel.login.state = state
    assert panel.quota_asks(config_module.gpu_option("a2-highgpu-1g")) == []


def test_a_region_with_quota_is_one_click_away(panel):
    panel.login.state = state_no_quota()
    option = config_module.gpu_option("a2-highgpu-1g")
    panel.use_region("us-west4", option)
    assert panel.login.profile.zone == "us-west4-b"


def test_a_region_we_have_no_zone_list_for_still_moves_the_profile(panel):
    panel.login.state = state_no_quota()
    option = config_module.gpu_option("a2-ultragpu-1g")
    panel.use_region("europe-west4", option)
    assert panel.login.profile.zone == "europe-west4-a"


def test_the_quota_card_draws_the_request_block(panel, context, render):
    panel.login.state = state_no_quota()
    render(panel._draw_machine, panel._draw_run)


def test_without_the_quotas_api_the_card_offers_to_turn_it_on(panel, context, render):
    state = state_no_quota()
    state.quota_infos = {}
    state.apis[account.SERVICE_QUOTAS] = False
    panel.login.state = state
    render(panel._draw_machine)


def test_asking_queues_the_requests_for_the_worker_thread(panel):
    started = []
    panel.login._start = lambda task: started.append(task.__name__)
    panel.login.state = state_no_quota()
    asks = panel.quota_asks(config_module.gpu_option("a2-highgpu-1g"))
    panel.login.ask_for_quota(asks)
    assert started == ["_task_quota"]
    assert panel.login.quota_asks == asks


def test_enabling_one_api_names_it_for_the_worker_thread(panel):
    started = []
    panel.login._start = lambda task: started.append(task.__name__)
    panel.login.enable_api(account.SERVICE_IAM)
    assert panel.login.api_pending == account.SERVICE_IAM
    assert started == ["_task_enable_api"]
