"""One runnable check of the thing that must not break: the learning loop.

python -m tests.test_loop     (or: pytest tests/test_loop.py)
"""
import os

os.environ.setdefault("ECHOLOOP_TRACE", "0")
os.environ.setdefault("ECHOLOOP_TYPESAFE", "0")
os.environ.setdefault("ECHOLOOP_LLM", "0")

from backend import memory, pipeline  # noqa: E402
from backend.bandit import Policy, initial_weights  # noqa: E402
from backend.schemas import FeedbackReq, ProcessReq, SupportProfile  # noqa: E402

SCENE = ["notebook", "headphones"]


def _process(con, text, pointing=None):
    return pipeline.process(con, ProcessReq(user_id="t", transcript=text,
                                            scene_hint=SCENE, pointing_hint=pointing))


def test_correction_changes_ranking():
    con = memory.connect(":memory:")
    memory.set_profile(con, "t", SupportProfile(support_mode="autism_neurodivergent",
                                                camera_enabled=True))
    first = _process(con, "I need... blue...")
    assert len(first["candidates"]) >= 2, "must always offer multiple hypotheses"
    assert first["none_fit_available"]

    fb = pipeline.feedback(con, FeedbackReq(interaction_id=first["interaction_id"],
                                            accepted=False,
                                            confirmed_text="I need my headphones."))
    assert fb["reward"] == -1
    assert fb["reflection"]["failure_type"] in ("RANKING_ERROR", "MISSING_CANDIDATE",
                                                "WRONG_POINTING_TARGET", "STALE_MEMORY")

    second = _process(con, "Can you get... blue thing...")
    assert second["top_candidate"] == "I need my headphones.", (
        f"correction did not change the ranking: {second['candidates']}")
    assert second["memory_matches"], "confirmed correction was not retrieved"


def test_accept_reinforces():
    con = memory.connect(":memory:")
    memory.set_profile(con, "t", SupportProfile(camera_enabled=True))
    r = _process(con, "I need... blue...")
    top = r["top_candidate"]
    fb = pipeline.feedback(con, FeedbackReq(interaction_id=r["interaction_id"],
                                            accepted=True, confirmed_text=top))
    assert fb["reward"] == 1
    assert fb["reflection"]["failure_type"] == "SUCCESS"
    assert fb["speak"] == top, "only confirmed text is offered for speech"


def test_none_of_these_is_never_spoken():
    con = memory.connect(":memory:")
    r = _process(con, "I need... blue...")
    fb = pipeline.feedback(con, FeedbackReq(interaction_id=r["interaction_id"],
                                            accepted=False, none_fit=True))
    assert fb["speak"] is None, "an unconfirmed guess must never be speakable"
    assert fb["reward"] == -1


def test_camera_off_means_no_visual_context():
    con = memory.connect(":memory:")
    memory.set_profile(con, "t", SupportProfile(camera_enabled=False))
    r = _process(con, "I need... blue...", pointing="headphones")
    assert r["perception"]["objects"] == []
    assert r["perception"]["gesture"]["target"] is None


def test_bandit_advantage_update_reorders():
    p = Policy(initial_weights())
    good = {"base": .3, "memory": .9, "visual": .2, "bias": 1.0}
    bad = {"base": .5, "memory": .1, "visual": .8, "bias": 1.0}
    mean = {k: (good[k] + bad[k]) / 2 for k in good}
    before = p.score(bad) - p.score(good)
    p.update(bad, -1, baseline=mean)     # we showed `bad`, user rejected it
    after = p.score(bad) - p.score(good)
    assert after < before, "negative reward must push the shown candidate down"


def test_forget_removes_memory():
    con = memory.connect(":memory:")
    memory.add_memory(con, "t", "blue", "desk", "I need my headphones.", 1)
    assert memory.search(con, "t", "blue")
    assert memory.forget(con, "t", "I need my headphones.") == 1
    assert memory.search(con, "t", "blue") == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
