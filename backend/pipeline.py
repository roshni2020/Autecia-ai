"""Agent-to-agent orchestration (spec §7). Shared by the API and the evaluator."""
import time
import uuid

from . import bandit, memory
from .agents import intent as intent_agent
from .agents import learning as learning_agent
from .agents import perception as perception_agent
from .agents import reflection as reflection_agent
from .integrations import log_trace
from .schemas import AgentMessage, FeedbackReq, ProcessReq


def _msg(mtype, frm, to, payload) -> dict:
    return AgentMessage(message_type=mtype, **{"from": frm, "to": to},
                        payload=payload).model_dump(by_alias=True)


def process(con, req: ProcessReq) -> dict:
    t0 = time.perf_counter()
    profile = memory.get_profile(con, req.user_id)
    policy = bandit.load(con, req.user_id, profile.support_mode)

    perc = perception_agent.run(req, profile.camera_enabled)
    query = f"{perc.transcript} {perception_agent.visual_summary(perc)}"
    memories = memory.search(con, req.user_id, query, k=3)
    intents = intent_agent.run(perc, memories, profile.suggestion_count)
    ranked = learning_agent.rerank(intents, perc, memories, policy)

    interaction_id = uuid.uuid4().hex[:12]
    session_id = req.session_id or uuid.uuid4().hex[:8]
    latency_ms = int((time.perf_counter() - t0) * 1000)

    bus = [
        _msg("perception_summary", "perception_agent", "intent_agent", perc.model_dump()),
        _msg("candidate_set", "intent_agent", "learning_agent",
             {"candidates": [c.model_dump() for c in intents.candidates]}),
        _msg("ranked_candidates", "learning_agent", "reflection_agent",
             {"ranked": [{"text": c.text, "score": c.score} for c in ranked],
              "policy_version": policy.version}),
    ]

    observation = {           # pre-decision record only (spec §14)
        "interaction_id": interaction_id, "user_id": req.user_id,
        "transcript": perc.transcript, "pause_intervals": perc.pause_intervals,
        "objects": [o.model_dump() for o in perc.objects],
        "pointing_target": perc.gesture.target,
        "perception": perc.model_dump(),
        "visual_summary": perception_agent.visual_summary(perc),
        "candidates": [c.model_dump() for c in ranked],
        "memory_matches": [m.model_dump() for m in memories],
        "policy_version": policy.version, "policy_weights": policy.as_dict(),
        "latency_ms": latency_ms,
    }
    memory.save_observation(con, interaction_id, req.user_id, session_id,
                            observation, policy.version)

    trace_summary = [
        f"Perception: {perception_agent.summary(perc)}",
        f"Intent: generated {len(ranked)} candidate meanings",
        f"Learning: {learning_agent.summary(memories, ranked)}",
    ]
    log_trace("interaction.process", {**observation, "trace_summary": trace_summary})

    return {"interaction_id": interaction_id, "session_id": session_id,
            "transcript": perc.transcript,
            "candidates": [c.text for c in ranked],
            "candidate_detail": [c.model_dump() for c in ranked],
            "top_candidate": ranked[0].text if ranked else None,
            "memory_matches": [m.model_dump() for m in memories],
            "trace_summary": trace_summary, "agent_messages": bus,
            "policy_version": policy.version, "policy_weights": policy.as_dict(),
            "perception": perc.model_dump(), "latency_ms": latency_ms,
            "none_fit_available": True}


def feedback(con, fb: FeedbackReq, learn: bool = True, remember: bool = True) -> dict:
    row = memory.get_interaction(con, fb.interaction_id)
    if row is None:
        raise KeyError(fb.interaction_id)
    import json
    obs = json.loads(row["observation"])
    user_id = row["user_id"]
    profile = memory.get_profile(con, user_id)
    policy = bandit.load(con, user_id, profile.support_mode)
    before = policy.as_dict()

    cands = obs["candidates"]
    shown = cands[0]
    reward = 1 if fb.accepted else -1
    confirmed = (fb.confirmed_text or (shown["text"] if fb.accepted else None) or "").strip()

    in_set = any(c["text"].strip().lower() == confirmed.lower() for c in cands)
    # A ranking policy can only learn from turns where the right meaning was on the
    # list. If it was never generated that is a generator miss (MISSING_CANDIDATE),
    # and updating the ranker on it is pure noise.
    if learn and (fb.accepted or in_set):
        # baseline = mean feature vector of the candidate set shown to the user
        keys = shown["features"].keys()
        mean = {k: sum(c["features"][k] for c in cands) / len(cands) for k in keys}
        # 1. reward update for the action we actually took
        policy.update(shown["features"], reward, baseline=mean)
        # 2. supervised preference step toward what the user really meant
        chosen = next((c for c in cands if c["id"] == fb.chosen_candidate_id
                       or c["text"].strip().lower() == confirmed.lower()), None)
        if not fb.accepted and chosen:
            policy.prefer(chosen["features"], shown["features"])
        bandit.save(con, user_id, policy)

    if confirmed and remember:
        memory.add_memory(con, user_id, obs["transcript"], obs["visual_summary"],
                          confirmed, reward)

    from .schemas import Perception
    refl = reflection_agent.run(shown["text"], confirmed, fb.accepted, fb.none_fit,
                                Perception(**obs["perception"]), cands,
                                obs["memory_matches"]).model_dump()

    record = {"interaction_id": fb.interaction_id, "accepted": fb.accepted,
              "rejected": not fb.accepted, "confirmed_text": confirmed, "reward": reward,
              "none_fit": fb.none_fit, "clarification_turns": 0 if fb.accepted else 1,
              "elapsed_ms": fb.elapsed_ms}
    memory.save_feedback(con, fb.interaction_id, record, refl)
    log_trace("interaction.feedback", {**record, "reflection": refl,
                                       "policy_before": before, "policy_after": policy.as_dict(),
                                       "policy_version": policy.version, "user_id": user_id})

    bus = [
        _msg("user_feedback", "user", "learning_agent",
             {"accepted": fb.accepted, "confirmed_text": confirmed, "reward": reward,
              "none_fit": fb.none_fit}),
        _msg("policy_update", "learning_agent", "reflection_agent",
             {"shown": shown["text"], "confirmed_text": confirmed,
              "weights_before": before, "weights_after": policy.as_dict(),
              "policy_version": policy.version, "memory_saved": bool(confirmed)}),
        _msg("reflection_result", "reflection_agent", "learning_agent", refl),
    ]
    return {"reward": reward, "confirmed_text": confirmed, "reflection": refl,
            "agent_messages": bus,
            "policy_version": policy.version, "policy_weights": policy.as_dict(),
            "policy_before": before, "memory_saved": bool(confirmed),
            "speak": confirmed if confirmed else None,
            "trace_summary": [f"Reflection: {refl['failure_type']} — {refl['reason_code']}"]}
