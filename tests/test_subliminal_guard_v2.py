"""subliminal_guard_v2.py: channel_scope on every audit() result (CHANNEL_SCOPE.md sections 1, 2, 4)."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "docs" / "research" / "subliminal_guard_v2.py"


@pytest.fixture(scope="module")
def sg():
    spec = importlib.util.spec_from_file_location("subliminal_guard_v2", str(PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ASK = lambda p: "owl"
PROMPTS = ["a", "b", "c"]


def test_self_test_passes(sg):
    assert sg.self_test() == 0


@pytest.mark.parametrize("blind", [[], ["semantic"], ["semantic", "activation"]])
def test_body_cannot_be_removed_by_an_argument(sg, blind):
    with pytest.raises(sg.ChannelScopeError):
        sg.audit("owl", ASK, PROMPTS, blind_channels=blind)


def test_body_in_blind_channels_of_every_result(sg):
    for regime in sg.REGIMES:
        r = sg.audit("owl", ASK, PROMPTS, regime=regime, blind_channels=["body", "x"])
        assert "body" in r["channel_scope"]["blind_channels"]
        assert set(r["channel_scope"]) == {"assumed_regime", "channels_reached",
                                           "blind_channels", "notes"}


def test_regime_out_of_reach_is_void_never_a_pass(sg):
    for regime in sg.REGIMES_OUT_OF_REACH:
        r = sg.audit("owl", ASK, PROMPTS, regime=regime)
        assert r["status"] == "VOID"
        assert regime in r["verdict"]
        assert "holds" not in r["verdict"]


def test_regime_outside_enum_rejected(sg):
    with pytest.raises(sg.ChannelScopeError):
        sg.audit("owl", ASK, PROMPTS, regime="surface")


def test_unknown_regime_pass_is_holds_regime_unknown_never_holds(sg):
    for kw in ({}, {"regime": "unknown"}):
        r = sg.audit("owl", ASK, PROMPTS, **kw)
        assert r["flags"] == []
        assert r["verdict"] == "holds_regime_unknown"
    assert sg.audit("owl", ASK, PROMPTS, regime="vocab_geometry")["verdict"].startswith("holds (")
