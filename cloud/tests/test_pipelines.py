"""How run parameters reach the pipeline function on the worker."""

from __future__ import annotations

import pytest

from imgui_cloud import pipelines


def test_masknmf_nests_leftover_params_under_settings():
    job = pipelines.build_job(
        pipelines.get("masknmf"),
        "/mnt/data/input",
        "/mnt/data/output",
        params={"planes": [1, 2], "registration_iterations": 5},
    )
    assert job["entry"] == "mbo_utilities.masknmf:run_volume"
    assert job["kwargs"]["input_data"] == "/mnt/data/input"
    assert job["kwargs"]["save_path"] == "/mnt/data/output"
    assert job["kwargs"]["planes"] == [1, 2]
    assert job["kwargs"]["settings"] == {"registration_iterations": 5}


def test_suite2p_spreads_params_as_keyword_arguments():
    job = pipelines.build_job(
        pipelines.get("suite2p"),
        "/mnt/data/input",
        "/mnt/data/output",
        params={"ops": {"diameter": 2}, "keep_reg": True},
    )
    assert "settings" not in job["kwargs"]
    assert job["kwargs"]["ops"] == {"diameter": 2}
    assert job["kwargs"]["keep_reg"] is True
    assert job["kwargs"]["keep_raw"] is False


def test_run_params_win_over_pipeline_defaults():
    job = pipelines.build_job(pipelines.get("suite2p"), "in", "out", {"keep_raw": True})
    assert job["kwargs"]["keep_raw"] is True


def test_unknown_pipeline_names_the_alternatives():
    with pytest.raises(KeyError, match="masknmf"):
        pipelines.get("does-not-exist")


def test_requirements_append_the_runs_own_pins():
    spec = pipelines.get("masknmf")
    reqs = pipelines.requirements(spec, ["git+https://example/repo@branch"])
    assert reqs[: len(spec.pip)] == spec.pip
    assert reqs[-1] == "git+https://example/repo@branch"


def test_register_adds_a_pipeline_without_editing_the_package():
    spec = pipelines.PipelineSpec(
        name="unit-test-pipeline", entry="mypkg:main", pip=["mypkg"]
    )
    pipelines.register(spec)
    try:
        assert "unit-test-pipeline" in pipelines.available()
        job = pipelines.build_job(pipelines.get("unit-test-pipeline"), "in", "out")
        assert job["entry"] == "mypkg:main"
    finally:
        pipelines._BUILTINS.pop("unit-test-pipeline")
