"""
Tests for canonical effective-frame-rate resolution.

resolve_effective_rate picks one authoritative rate from a metadata dict
that may carry stale aliases stamped by older writers, and get_param
delegates fs/finterval lookups to it.
"""

import logging

import pytest

from mbo_utilities.metadata import params
from mbo_utilities.metadata.params import get_param, resolve_effective_rate


@pytest.fixture(autouse=True)
def _fresh_stale_warning_cache():
    """each test sees the stale-alias warning dedupe cache empty."""
    params._STALE_WARNED.clear()
    yield
    params._STALE_WARNED.clear()


# alias state of a real x4-decimated store: the writer updated only
# fs/frame_rate/finterval, leaving every other registered alias at the
# source rate (19.66 Hz)
DECIMATED = {
    "fs": 4.915,
    "frame_rate": 4.915,
    "finterval": 0.2034588,
    "framerate": 19.66,
    "fr": 19.66,
    "frameRate": 19.66,
    "scanFrameRate": 19.66,
    "fps": 19.66,
    "sampling_frequency": 19.66,
    "dt": 0.0508647,
    "frame_interval": 0.0508647,
    "FrameInterval": 0.0508647,
    "time_interval": 0.0508647,
}


def _ome_dict(time_scale: float) -> dict:
    return {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        {"name": "t", "type": "time", "unit": "second"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale",
                                 "scale": [time_scale, 5.0, 1.0, 1.0]}
                            ],
                        }
                    ],
                }
            ],
        }
    }


def _rate_warnings(caplog):
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "stale frame-rate" in r.getMessage()
    ]


class TestResolveEffectiveRate:
    def test_decimated_store_resolves_canonical_fs(self, caplog):
        """the real decimated dict resolves from canonical fs, warning once
        about the stale aliases."""
        with caplog.at_level(logging.WARNING, logger="mbo_utilities"):
            fs, finterval, source = resolve_effective_rate(dict(DECIMATED))

        assert fs == pytest.approx(4.915, rel=1e-3)
        assert finterval == pytest.approx(1.0 / 4.915, rel=1e-3)
        assert source == "fs"
        assert len(_rate_warnings(caplog)) == 1

    def test_finterval_only_resolves_via_finterval(self):
        """a dict with only finterval still resolves through it."""
        fs, finterval, source = resolve_effective_rate({"finterval": 0.2})
        assert fs == pytest.approx(5.0)
        assert finterval == pytest.approx(0.2)
        assert source == "finterval"

    def test_fps_only(self):
        """a lone low-precedence alias still resolves."""
        fs, finterval, source = resolve_effective_rate({"fps": 19.66})
        assert fs == pytest.approx(19.66)
        assert finterval == pytest.approx(1.0 / 19.66)
        assert source == "fps"

    def test_ome_multiscales_time_scale(self):
        """OME per-level time scale resolves when no key-based rate exists."""
        fs, finterval, source = resolve_effective_rate(_ome_dict(0.2034588))
        assert fs == pytest.approx(4.915, rel=1e-3)
        assert finterval == pytest.approx(0.2034588)
        assert source == "ome.multiscales"

    def test_top_level_multiscales(self):
        """a top-level multiscales attr is accepted too."""
        md = _ome_dict(0.2034588)["ome"]
        del md["version"]
        fs, _, source = resolve_effective_rate(md)
        assert fs == pytest.approx(4.915, rel=1e-3)
        assert source == "multiscales"

    def test_ome_time_scale_key(self):
        """the _ome_time_scale key stamped by the zarr reader resolves."""
        fs, finterval, _ = resolve_effective_rate({"_ome_time_scale": 0.2034588})
        assert fs == pytest.approx(4.915, rel=1e-3)
        assert finterval == pytest.approx(0.2034588)

    def test_ome_scale_of_one_is_unset(self):
        """scale 1.0 is the writer default, not a real rate."""
        assert resolve_effective_rate(_ome_dict(1.0)) == (None, None, "")
        assert resolve_effective_rate({"_ome_time_scale": 1.0}) == (None, None, "")

    def test_none_canonicals_block_stale_alias(self):
        """nulled canonicals (non-contiguous selection) must not fall
        through to a stale alias."""
        md = {"fs": None, "frame_rate": None, "finterval": None,
              "framerate": 19.66}
        assert resolve_effective_rate(md) == (None, None, "")

    def test_unregistered_case_variant(self):
        """case-insensitive matching picks up unregistered spellings."""
        fs, _, source = resolve_effective_rate({"Framerate": 20})
        assert fs == pytest.approx(20.0)
        assert source == "Framerate"

    def test_case_colliding_agreement_no_warning(self, caplog):
        """two case-variants of one alias that agree produce no warning."""
        with caplog.at_level(logging.WARNING, logger="mbo_utilities"):
            fs, _, _ = resolve_effective_rate(
                {"framerate": 20.0, "frameRate": 20.0}
            )
        assert fs == pytest.approx(20.0)
        assert _rate_warnings(caplog) == []

    def test_case_colliding_divergence_warns(self, caplog):
        """diverging case-variants produce one warning naming both."""
        with caplog.at_level(logging.WARNING, logger="mbo_utilities"):
            fs, _, _ = resolve_effective_rate(
                {"framerate": 20.0, "frameRate": 25.0}
            )
        assert fs == pytest.approx(20.0)
        warnings = _rate_warnings(caplog)
        assert len(warnings) == 1
        assert "frameRate" in warnings[0].getMessage()

    def test_exact_spelling_outranks_case_variant(self, caplog):
        """the registered spelling wins a same-rank tie regardless of
        dict insertion order."""
        with caplog.at_level(logging.WARNING, logger="mbo_utilities"):
            fs, finterval, source = resolve_effective_rate(
                {"FS": 10.0, "fs": 20.0}
            )
        assert fs == pytest.approx(20.0)
        assert finterval == pytest.approx(0.05)
        assert source == "fs"
        warnings = _rate_warnings(caplog)
        assert len(warnings) == 1
        assert "FS" in warnings[0].getMessage()

    def test_stale_warning_deduped_across_resolves(self, caplog):
        """repeat resolves of the same divergent dict warn exactly once
        (hot render loops call get_param(md, 'fs') per frame)."""
        md = dict(DECIMATED)
        with caplog.at_level(logging.WARNING, logger="mbo_utilities"):
            first = resolve_effective_rate(md)
            second = resolve_effective_rate(md)

        assert first == second
        assert len(_rate_warnings(caplog)) == 1

    def test_empty_and_none(self):
        assert resolve_effective_rate({}) == (None, None, "")
        assert resolve_effective_rate(None) == (None, None, "")

    def test_skips_non_numeric_and_zero(self):
        md = {"fs": "not-a-number", "framerate": 0, "fps": 12.5}
        fs, _, source = resolve_effective_rate(md)
        assert fs == pytest.approx(12.5)
        assert source == "fps"


class TestGetParamDelegation:
    def test_get_param_fs_from_decimated(self):
        assert get_param(dict(DECIMATED), "fs") == pytest.approx(4.915, rel=1e-3)

    def test_get_param_finterval_from_decimated(self):
        assert get_param(dict(DECIMATED), "finterval") == pytest.approx(
            0.2034588, rel=1e-6
        )

    def test_nulled_canonicals_do_not_leak_stale_alias(self):
        md = {"fs": None, "frame_rate": None, "finterval": None,
              "framerate": 19.66}
        assert get_param(md, "fs") is None
        assert get_param(md, "finterval") is None

    def test_default_honored_when_nothing_resolves(self):
        assert get_param({}, "fs", default=3.0) == 3.0
        md = {"fs": None, "framerate": 19.66}
        assert get_param(md, "fs", default=None) is None
        assert get_param(md, "fs", default=7.5) == 7.5

    def test_override_still_wins(self):
        assert get_param(dict(DECIMATED), "fs", override=2.0) == 2.0
