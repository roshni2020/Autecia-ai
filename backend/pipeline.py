"""Agent-to-agent orchestration (spec §7). Shared by the API and the evaluator."""
import time
import uuid

from . import agent_trace, bandit, memory
from .agents import intent as intent_agent
from .agents import learning as learning_agent
from .agents import perception as perception_agent
from .agents import reflection as reflection_agent
from .integrations import log_trace
from .schemas import AgentMessage, FeedbackReq, ProcessReq
from .speech import analyze_audio
from .speech.audio import decode_data_url
from .speech.schemas import SpeechAnalysis


def _speech(req: ProcessReq) -> SpeechAnalysis | None:
    """Audio -> SpeechAnalysis. Precomputed features (datasets, eval) pass straight through."""
    if req.audio:
        try:
            data, suffix = decode_data_url(req.audio)
            return analyze_audio(data, suffix, transcript_hint=req.transcript)
        except Exception as e:
            print(f"[pipeline] speech analysis skipped: {e}")
            return None
    if req.speech_features:
        try:
            return SpeechAnalysis(**req.speech_features)
        except Exception as e:
            print(f"[pipeline] bad speech_features ignored: {e}")
    return None


def _msg(mtype, frm, to, payload) -> dict:
    return AgentMessage(message_type=mtype, **{"from": frm, "to": to},
                        payload=payload).model_dump(by_alias=True)


def process(con, req: ProcessReq) -> dict:
    t0 = time.perf_counter()
    profile = memory.get_profile(con, req.user_id)
    policy = bandit.load(con, req.user_id, profile.support_mode)

    interaction_id = uuid.uuid4().hex[:12]
    session_id = req.session_id or uuid.uuid4().hex[:8]

    analysis = _speech(req)
    obs = analysis.observations if analysis else None
    if obs and obs.transcript.strip() and obs.providers.get("asr", "client") != "client":
        req.transcript = obs.transcript          # server ASR is the primary path

    with agent_trace.turn(session_id, req.user_id, req.transcript, "suggest") as t:
        with agent_trace.subagent(t, "perception_agent", {"transcript": req.transcript,
                                  "scene": req.scene_hint, "pointing": req.pointing_hint,
                                  "speech": obs.model_dump(exclude={"words"}) if obs else None},
                                  "Observable facts from speech + scene") as span:
            perc = perception_agent.run(req, profile.camera_enabled, speech=obs)
            span["output"] = perc.model_dump(exclude={"speech"})
        query = f"{perc.transcript} {perception_agent.visual_summary(perc)}"
        utterance = analysis.fused_embedding if analysis else None
        memories = memory.search(con, req.user_id, query, k=3, utterance=utterance)
        with agent_trace.subagent(t, "intent_agent", {"perception": perception_agent.summary(perc),
                                  "memories": [m.confirmed_text for m in memories]},
                                  "2-4 candidate meanings + none-fit") as span:
            intents = intent_agent.run(perc, memories, profile.suggestion_count)
            span["output"] = [c.text for c in intents.candidates]
            from .integrations import LAST_LLM
            if LAST_LLM.get("output") is not None:
                agent_trace.llm_span(t, LAST_LLM["model"], LAST_LLM["provider"], LAST_LLM["prompt"],
                                     LAST_LLM["system"], LAST_LLM["output"], LAST_LLM["usage"])
        with agent_trace.subagent(t, "learning_agent", {"candidates": [c.text for c in intents.candidates],
                                  "policy_version": policy.version},
                                  "Memory retrieval + contextual-bandit reranking") as span:
            judgment = learning_agent.judge(perc, memories, [c.text for c in intents.candidates])
            if judgment:
                agent_trace.tool_span(t, "typesafe.system_one", {"question": "which candidate is meant"},
                                      judgment)
            ranked = learning_agent.rerank(intents, perc, memories, policy, judgment)
            span["output"] = [{"text": c.text, "score": c.score} for c in ranked]
        if t is not None:
            import weave
            t.record(output_messages=[weave.Message(role="assistant",
                                                    content=f"Do you mean: {ranked[0].text}" if ranked else "")])
    latency_ms = int((time.perf_counter() - t0) * 1000)

    bus = [
        _msg("perception_summary", "perception_agent", "intent_agent", perc.model_dump()),
        _msg("candidate_set", "intent_agent", "learning_agent",
             {"candidates": [c.model_dump() for c in intents.candidates]}),
        *([_msg("system_one_judgment", "typesafe", "learning_agent", judgment)] if judgment else []),
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
        "judgment": judgment,
        "speech": obs.model_dump(exclude={"words"}) if obs else None,
        "utterance_embedding": utterance,        # derived vector only; audio is gone
        "policy_version": policy.version, "policy_weights": policy.as_dict(),
        "latency_ms": latency_ms,
    }
    memory.save_observation(con, interaction_id, req.user_id, session_id,
                            observation, policy.version)

    trace_summary = [
        f"Perception: {perception_agent.summary(perc)}",
        f"Intent: generated {len(ranked)} candidate meanings via {intent_agent.LAST_PROVIDER['name']}",
        f"Learning: {learning_agent.summary(memories, ranked)}"
        + (f"; System One favours \"{judgment['choice']}\" "
           f"({int(judgment['confidence'] * 100)}% confidence)" if judgment else ""),
    ]
    log_trace("interaction.process", {**observation, "trace_summary": trace_summary})

    return {"interaction_id": interaction_id, "session_id": session_id,
            "transcript": perc.transcript,
            "candidates": [c.text for c in ranked],
            "candidate_detail": [c.model_dump() for c in ranked],
            "top_candidate": ranked[0].text if ranked else None,
            "memory_matches": [m.model_dump() for m in memories],
            "speech_observations": obs.model_dump() if obs else None,
            "judgment": judgment, "intent_provider": intent_agent.LAST_PROVIDER["name"],
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
        sp = obs.get("speech") or {}
        summary = (f"{sp['vad']['pause_count']} pause(s), "
                   f"{'fragmented' if sp.get('fragmented') else 'fluent'}") if sp else ""
        memory.add_memory(con, user_id, obs["transcript"], obs["visual_summary"],
                          confirmed, reward, utterance_embedding=obs.get("utterance_embedding"),
                          speech_summary=summary)

    from .schemas import Perception
    feedback_msg = ("accepted" if fb.accepted else "rejected") + (f": {confirmed}" if confirmed else " (none fit)")
    with agent_trace.turn(row["session_id"] or fb.interaction_id, user_id, feedback_msg, "feedback") as t:
        with agent_trace.subagent(t, "learning_agent", {"reward": reward, "shown": shown["text"],
                                  "confirmed": confirmed},
                                  "Reward + preference update of the per-user policy") as span:
            span["output"] = {"policy_before": before, "policy_after": policy.as_dict(),
                              "version": policy.version, "memory_saved": bool(confirmed)}
        with agent_trace.subagent(t, "reflection_agent", {"prediction": shown["text"],
                                  "confirmed": confirmed, "accepted": fb.accepted},
                                  "Why the shown suggestion succeeded or failed") as span:
            refl = reflection_agent.run(shown["text"], confirmed, fb.accepted, fb.none_fit,
                                        Perception(**obs["perception"]), cands,
                                        obs["memory_matches"]).model_dump()
            if refl.get("reason_code") == "SYSTEM_ONE_JUDGMENT":
                agent_trace.tool_span(t, "typesafe.system_one", {"question": "failure type"}, refl)
            span["output"] = refl
        if t is not None:
            import weave
            t.record(output_messages=[weave.Message(role="assistant",
                                                    content=f"{refl['failure_type']}: {refl['recommendation']}")])

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
