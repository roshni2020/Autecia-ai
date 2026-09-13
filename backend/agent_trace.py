"""Weave agent-observability spans for the four EchoLoop agents.

Maps EchoLoop onto Weave's agent model so the Agents dashboard shows it:
    session  -> Conversation (agent_name="echoloop")
    request  -> Turn (user message = raw transcript / feedback)
    agent    -> SubAgent span (perception_agent, intent_agent, learning_agent, reflection_agent)
    W&B Inference call -> LLM span with token usage
    TypeSafe System One -> Tool span

Everything here is a no-op when Weave is not live (no key / ECHOLOOP_TRACE=0).
"""
import contextlib
import json
import os
import time

_conversations: dict[str, object] = {}     # session_id -> weave Conversation
MAX_CONVERSATIONS = 500                     # ponytail: in-memory registry; sessions are short


def live() -> bool:
    from .integrations import TRACE_CALLS, weave_init
    return bool(TRACE_CALLS and os.getenv("WANDB_API_KEY") and weave_init())


def _conversation(session_id: str, user_id: str):
    import weave
    conv = _conversations.get(session_id)
    if conv is None:
        if len(_conversations) >= MAX_CONVERSATIONS:
            _conversations.pop(next(iter(_conversations)))
        conv = weave.start_conversation(
            agent_name="echoloop", conversation_id=f"{session_id}-{int(time.time() * 1000)}",
            conversation_name=f"{user_id} · {session_id}",
            attributes={"user_id": user_id, "app": "echoloop"})
        _conversations[session_id] = conv
    return conv


@contextlib.contextmanager
def turn(session_id: str, user_id: str, user_message: str, phase: str):
    """One request = one turn. Yields the Turn (or None when tracing is off)."""
    if not live():
        yield None
        return
    try:
        conv = _conversation(session_id, user_id)
        t = conv.start_turn(user_message=user_message, agent_name="echoloop")
        t.record(agent_description=f"EchoLoop {phase}: perception -> intent -> learning -> reflection")
    except Exception as e:
        print(f"[agent_trace] turn not started: {e}")
        yield None
        return
    try:
        yield t
    finally:
        with contextlib.suppress(Exception):
            t.end()


@contextlib.contextmanager
def subagent(t, name: str, inputs: dict | None = None, description: str = ""):
    """A named agent span inside the turn. `yield`s a dict you can put 'output' into."""
    out: dict = {}
    if t is None:
        yield out
        return
    import weave
    sa = None
    try:
        sa = t.start_subagent(name=name)
        sa.record(agent_description=description,
                  input_messages=[weave.Message(role="user", content=_short(inputs))] if inputs else None)
    except Exception as e:
        print(f"[agent_trace] subagent {name}: {e}")
    try:
        yield out
    finally:
        if sa is not None:
            with contextlib.suppress(Exception):
                if out.get("output") is not None:
                    sa.record(output_messages=[weave.Message(role="assistant", content=_short(out["output"]))])
                sa.end()


def llm_span(t, model: str, provider: str, prompt: str, system: str, output: str | None,
             usage: dict | None, latency_ms: int | None = None) -> None:
    """Record one LLM call (already completed) as an LLM span under the current turn."""
    if t is None:
        return
    try:
        import weave
        llm = weave.start_llm(model=model, provider_name=provider,
                              system_instructions=[system] if system else None)
        llm.record(input_messages=[weave.Message(role="user", content=prompt)],
                   output_messages=[weave.Message(role="assistant", content=output or "")],
                   usage=weave.Usage(input_tokens=int((usage or {}).get("prompt_tokens", 0)),
                                     output_tokens=int((usage or {}).get("completion_tokens", 0))),
                   response_model=model)
        llm.end()
    except Exception as e:
        print(f"[agent_trace] llm span: {e}")


def tool_span(t, name: str, arguments: dict, result) -> None:
    if t is None:
        return
    try:
        import weave
        tool = weave.start_tool(name=name, arguments=_short(arguments))
        tool.result = _short(result)
        tool.end()
    except Exception as e:
        print(f"[agent_trace] tool span: {e}")


def close(session_id: str) -> None:
    """End the session's conversation so the dashboard shows it; the next request
    of the same session opens a fresh one under the same conversation name."""
    conv = _conversations.pop(session_id, None)
    if conv is not None:
        with contextlib.suppress(Exception):
            conv.end()


def _short(x, limit: int = 4000) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= limit else s[:limit] + "…"
