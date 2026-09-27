import numpy as np
import pytest

from crashvlm.eval.events import GT, apply_constant, constant_baseline, evaluate
from crashvlm.models.probe import frame_targets


def gt(clip, test, present, t=None, timing_ok=True, event="airbag_deploy_start"):
    return GT(clip, test, "VTB", "occupant", event, present, t, timing_ok)


def test_presence_and_timing_metrics():
    gts = [
        gt("a", 1, True, 22.0),
        gt("b", 2, True, 22.0),
        gt("c", 3, True, 30.0),
        gt("d", 4, False),
        gt("e", 5, True, 14.0, timing_ok=False),  # counts for presence, not timing
    ]
    preds = {
        ("a", "airbag_deploy_start"): 22.0,  # exact
        ("b", "airbag_deploy_start"): 34.0,  # 12 ms off
        ("c", "airbag_deploy_start"): None,  # missed
        ("d", "airbag_deploy_start"): 40.0,  # false positive
        ("e", "airbag_deploy_start"): 90.0,  # detected; timing not scored
    }
    m = evaluate(gts, preds, tols=(8, 16), n_boot=50)["airbag_deploy_start"]
    assert m["precision"] == pytest.approx(3 / 4) and m["recall"] == pytest.approx(3 / 4)
    assert m["n_timing"] == 3
    assert m["hit@8"] == pytest.approx(1 / 3)  # the miss counts as a miss
    assert m["hit@16"] == pytest.approx(2 / 3)
    assert m["median_abs_err_ms"] == pytest.approx(6.0)
    lo, hi = m["f1_ci"]
    assert 0 <= lo <= m["f1"] <= hi <= 1


def test_constant_baseline_uses_train_median_and_prevalence():
    train = [gt("a", 1, True, 22.0), gt("b", 2, True, 30.0), gt("c", 3, True, 22.0)]
    train += [gt(f"x{i}", 10 + i, False, event="door_opens") for i in range(3)]
    train += [gt("y", 20, True, 60.0, event="door_opens")]
    table = constant_baseline(train)
    assert table[("airbag_deploy_start", "VTB", "occupant")] == 22.0
    assert table[("door_opens", "VTB", "occupant")] is None  # present in < half of clips
    p = apply_constant(table, [gt("t", 99, True, 22.0)])
    assert p[("t", "airbag_deploy_start")] == 22.0


def test_frame_targets():
    t = np.array([-10, -2, 6, 14, 22, 30], dtype=float)
    assert frame_targets(t, 14.0).tolist() == [0, 0, 0, 1, 1, 1]
    assert frame_targets(t, None).tolist() == [0] * 6
