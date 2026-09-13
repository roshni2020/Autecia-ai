"""Weave Evaluation of EchoLoop's intent recovery (shows up in the Weave Evals tab).

    python -m eval.weave_eval --rows 60 [--llm 0]

Each row: fragment + visible objects + pointing -> candidates -> top suggestion.
Scorers compare to the confirmed meaning. Rows are scored independently (Weave
Evaluation semantics), so this measures the *frozen* system; personalisation
over time is measured by eval/run_eval.py's chronological replay.
"""
import argparse
import asyncio
import csv
import os

os.environ.setdefault("ECHOLOOP_TRACE", "0")      # op spans off; the Evaluation logs itself
os.environ.setdefault("ECHOLOOP_TYPESAFE", "0")

from backend.integrations import load_env  # noqa: E402

load_env()

import weave  # noqa: E402

from backend import memory, pipeline  # noqa: E402
from backend.schemas import ProcessReq, SupportProfile  # noqa: E402
from eval.run_eval import CSV_PATH  # noqa: E402


class EchoLoopModel(weave.Model):
    """Frozen EchoLoop: perception -> intent -> learning (memory + bandit) -> top suggestion."""
    llm: bool = True
    camera: bool = True

    @weave.op
    def predict(self, fragment: str, visible_objects: str, pointing_target: str) -> dict:
        con = memory.connect(":memory:")                 # no memory carry-over between rows
        memory.set_profile(con, "eval", SupportProfile(camera_enabled=self.camera))
        r = pipeline.process(con, ProcessReq(
            user_id="eval", transcript=fragment,
            scene_hint=[o for o in visible_objects.split(";") if o] if self.camera else [],
            pointing_hint=(pointing_target or None) if self.camera else None))
        return {"top": r["top_candidate"], "candidates": r["candidates"],
                "provider": r["intent_provider"], "latency_ms": r["latency_ms"]}


def _norm(s: str) -> str:
    return " ".join(w for w in "".join(c if c.isalnum() or c == " " else " " for c in s.lower()).split()
                    if w not in {"the", "a", "an", "my", "to", "please"})


@weave.op
def top1(confirmed_text: str, output: dict) -> dict:
    return {"top1": _norm(output["top"] or "") == _norm(confirmed_text)}


@weave.op
def top3(confirmed_text: str, output: dict) -> dict:
    return {"top3": any(_norm(c) == _norm(confirmed_text) for c in output["candidates"])}


@weave.op
def candidate_count(output: dict) -> dict:
    n = len(output["candidates"])
    return {"has_multiple": n >= 2, "n_candidates": n}


@weave.op
def latency(output: dict) -> dict:
    return {"latency_ms": output["latency_ms"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--llm", type=int, default=1, help="1 = W&B Inference candidates, 0 = templates")
    a = ap.parse_args()
    if not a.llm:
        os.environ["ECHOLOOP_LLM"] = "0"

    weave.init(f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}")
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"][: a.rows]
    dataset = weave.Dataset(name="echoloop-synthetic-test", rows=[
        {"fragment": r["fragment"], "visible_objects": r["visible_objects"],
         "pointing_target": r["pointing_target"], "confirmed_text": r["confirmed_text"]} for r in rows])

    evaluation = weave.Evaluation(name="echoloop-intent-recovery", dataset=dataset,
                                  scorers=[top1, top3, candidate_count, latency])
    result = asyncio.run(evaluation.evaluate(EchoLoopModel(llm=bool(a.llm))))
    print({k: v for k, v in result.items() if k in ("top1", "top3", "candidate_count", "latency")})


if __name__ == "__main__":
    main()
