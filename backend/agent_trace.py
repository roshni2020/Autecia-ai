"""Weave Agents dashboard spans for the four EchoLoop agents (OpenTelemetry GenAI).

Weave's Agents/Conversations views are built from OTel spans carrying the GenAI
semantic-convention attributes. We emit them directly to the agents OTLP
endpoint — the same wire format the Weave conversation SDK uses — because that
is what verifiably registers agents for this project.

    session    -> gen_ai.conversation.id shared by every span
    agent step -> span "invoke_agent <agent>"  (perception_agent, intent_agent,
                  learning_agent, reflection_agent) with input/output messages
    LLM call   -> child span "chat <model>" with token usage
    TypeSafe   -> child span "execute_tool typesafe.system_one"

No-op without a W&B key or with ECHOLOOP_TRACE=0.
"""
import base64
import contextlib
import json
import os
from functools import lru_cache

ENDPOINT = "https://trace.wandb.ai/agents/otel/v1/traces"

DESCRIPTIONS = {
    "perception_agent": "Observable facts from speech + scene (transcript, pauses, objects, pointing)",
    "intent_agent": "2-4 candidate meanings + none-fit (W&B Inference / templates)",
    "learning_agent": "Memory retrieval, TypeSafe judgment, contextual-bandit reranking and policy update",
    "reflection_agent": "Why the shown suggestion succeeded or failed",
}


def live() -> bool:
    from .integrations import TRACE_CALLS
    return bool(TRACE_CALLS and os.getenv("WANDB_API_KEY") and os.getenv("WANDB_ENTITY")
                and os.getenv("WANDB_PROJECT"))


@lru_cache(maxsize=1)
def _provider():
    """Own TracerProvider -> batch OTLP export. Deliberately not the global provider."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    auth = "Basic " + base64.b64encode(f"api:{os.environ['WANDB_API_KEY']}".encode()).decode()
    project_id = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"   # raw; spaces are fine
    exporter = OTLPSpanExporter(endpoint=ENDPOINT, headers={"Authorization": auth, "project_id": project_id})
    provider = TracerProvider(resource=Resource.create({"service.name": "echoloop"}))
    provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=2000))
    return provider


def _tracer():
    return _provider().get_tracer("echoloop.agents")


def _msgs(role: str, content) -> str:
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return json.dumps([{"role": role, "parts": [{"type": "text", "content": text[:6000]}]}])


class _Session:
    def __init__(self, session_id: str, user_id: str):
        self.conversation_id = session_id
        self.user_id = user_id


def session(session_id: str, user_id: str):
    return _Session(session_id, user_id) if live() else None


@contextlib.contextmanager
def turn(sess, agent: str, user_message: str, inputs: dict | None = None):
    """One agent step. Yields a dict; set out["output"] and it becomes the output message."""
    out: dict = {}
    if sess is None:
        yield out
        return
    try:
        from opentelemetry import trace as otel
        messages = [{"role": "user", "parts": [{"type": "text", "content": user_message}]}]
        if inputs:
            messages.append({"role": "system", "parts": [{"type": "text",
                             "content": json.dumps(inputs, default=str)[:6000]}]})
        span = _tracer().start_span(f"invoke_agent {agent}", attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": agent,
            "gen_ai.agent.id": agent,
            "gen_ai.agent.description": DESCRIPTIONS.get(agent, ""),
            "gen_ai.conversation.id": sess.conversation_id,
            "gen_ai.provider.name": "echoloop",
            "user.id": sess.user_id,
            "gen_ai.input.messages": json.dumps(messages),
        })
        out["turn"] = span
        out["_ctx"] = otel.set_span_in_context(span)
    except Exception as e:
        print(f"[agent_trace] span {agent}: {e}")
        yield out
        return
    try:
        yield out
    finally:
        with contextlib.suppress(Exception):
            if out.get("output") is not None:
                span.set_attribute("gen_ai.output.messages", _msgs("assistant", out["output"]))
            span.end()


def llm_span(parent: dict | None, model: str, provider: str, prompt: str, system: str,
             output: str | None, usage: dict | None, latency_ms: int | None = None) -> None:
    if not parent or "turn" not in parent:
        return
    try:
        span = _tracer().start_span(f"chat {model}", context=parent["_ctx"], attributes={
            "gen_ai.operation.name": "chat", "gen_ai.provider.name": provider,
            "gen_ai.request.model": model, "gen_ai.response.model": model,
            "gen_ai.usage.input_tokens": int((usage or {}).get("prompt_tokens", 0)),
            "gen_ai.usage.output_tokens": int((usage or {}).get("completion_tokens", 0)),
            "gen_ai.system_instructions": json.dumps([{"type": "text", "content": system}]) if system else "",
            "gen_ai.input.messages": _msgs("user", prompt),
            "gen_ai.output.messages": _msgs("assistant", output or ""),
        })
        span.end()
    except Exception as e:
        print(f"[agent_trace] llm span: {e}")


def tool_span(parent: dict | None, name: str, arguments: dict, result) -> None:
    if not parent or "turn" not in parent:
        return
    try:
        span = _tracer().start_span(f"execute_tool {name}", context=parent["_ctx"], attributes={
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.arguments": json.dumps(arguments, default=str)[:4000],
            "gen_ai.tool.call.result": json.dumps(result, default=str)[:4000],
        })
        span.end()
    except Exception as e:
        print(f"[agent_trace] tool span: {e}")


def close(session_id: str) -> None:
    """Flush after the feedback step so a demo interaction shows up promptly."""
    if live():
        with contextlib.suppress(Exception):
            _provider().force_flush(5000)
