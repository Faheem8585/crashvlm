from crashvlm.eval.baselines import Blind
from crashvlm.eval.events import GT


def _gt(cid, t):
    return GT(cid, int(cid[1:]), "VTB", "occupant", "airbag_deploy_start", True, t, True)


def test_blind_prior_uses_training_crops():
    # every clip starts at -40 ms and the airbag fires at +20 ms: 60 ms after the clip start
    train = [_gt(f"c{i}", 20.0) for i in range(5)]
    start = {g.clip_id: -40.0 for g in train} | {"c9": -40.0}
    crop = {g.clip_id: 20.0 for g in train}  # training clips cropped by 20 ms
    test = [_gt("c9", 20.0)]
    # the test clip is cropped by 20 ms too, so a fair prior lands exactly on the event
    fair = Blind(train, start, crop).predict(test, start, {"c9": 20.0})
    assert fair[("c9", "airbag_deploy_start")] == 20.0
    # without training crops the prior is late by the crop
    biased = Blind(train, start).predict(test, start, {"c9": 20.0})
    assert biased[("c9", "airbag_deploy_start")] == 40.0
