"""2x2 effect arithmetic and the item bootstrap."""
import pytest

from lerp.evolution.factorial import bootstrap, effects


def test_effects_recover_known_main_effects_and_interaction():
    # base 0.10; keeping optimizer state is worth +0.04, one global schedule +0.02, no interaction
    c2, a, b = 0.10, 0.14, 0.12
    c1 = c2 + 0.04 + 0.02
    e = effects(c1, a, b, c2)
    assert e["state"] == pytest.approx(0.04) and e["schedule"] == pytest.approx(0.02)
    assert e["interaction"] == pytest.approx(0.0) and e["restart_cost"] == pytest.approx(0.06)
    # a synergy: only both together help
    e = effects(c1=0.20, a=0.10, b=0.10, c2=0.10)
    assert e["state"] == pytest.approx(0.05) and e["schedule"] == pytest.approx(0.05) and e["interaction"] == pytest.approx(0.10)
    # only the state matters
    e = effects(c1=0.14, a=0.14, b=0.10, c2=0.10)
    assert e["state"] == pytest.approx(0.04) and e["schedule"] == pytest.approx(0.0)


def _cells(n, rates, offset):
    # deterministic 0/1 patterns with the requested accuracies
    return {k: [1 if (i * 7 + offset) % 100 < round(r * 100) else 0 for i in range(n)] for k, r in rates.items()}


def test_bootstrap_interval_covers_the_effect_and_flags_a_zero_effect():
    n = 1000
    big = [_cells(n, {"c1": .30, "a": .20, "b": .20, "c2": .10}, o) for o in (0, 13, 29)]
    out = bootstrap(big, resamples=300, seed=1)
    assert out["mean"]["restart_cost"] == pytest.approx(0.20, abs=0.02)
    lo, hi = out["interval"]["restart_cost"]
    assert lo < out["mean"]["restart_cost"] < hi and lo > 0.1
    same = [_cells(n, {"c1": .20, "a": .20, "b": .20, "c2": .20}, o) for o in (0, 13, 29)]
    out = bootstrap(same, resamples=300, seed=1)
    lo, hi = out["interval"]["state"]
    assert abs(out["mean"]["state"]) < 0.01 and lo <= 0.0 <= hi
    with pytest.raises(ValueError):
        bootstrap([{"c1": [1, 0], "a": [1], "b": [1, 0], "c2": [1, 0]}])
