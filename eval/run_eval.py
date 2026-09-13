"""Offline evaluation + ablations on the synthetic environment (spec §12, §24).

Replays interactions chronologically per user through the real agent pipeline.
No later feedback ever reaches an earlier prediction.

Usage: python -m eval.run_eval [--limit N]
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("ECHOLOOP_TRACE", "0")
os.environ.setdefault("ECHOLOOP_TYPESAFE", "0")
os.environ.setdefault("ECHOLOOP_LLM", "0")  # batch replay: no per-call spans

from backend import memory, pipeline  # noqa: E402
from backend.integrations import log_trace, status, weave_init  # noqa: E402
from backend.schemas import FeedbackReq, ProcessReq, SupportProfile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "EchoLoop_5000_Synthetic_Interactions.csv"
REPORT = ROOT / "data" / "evaluation_report.json"

# Speech-feature ablations only differ when rows carry audio features (an
# --audio-dataset produced by data/preprocess_audio_dataset.py). On the synthetic
# CSV they collapse onto "transcript only" and are reported as such.
ABLATIONS = [
    dict(name="1. speech transcript only", video=False, mem=False, rl=False, branches=("text",)),
    dict(name="2. speech + GeMAPS", video=False, mem=True, rl=True, branches=("text", "gemaps")),
    dict(name="3. speech + WavLM", video=False, mem=True, rl=True, branches=("text", "speech")),
    dict(name="4. speech + WavLM + GeMAPS", video=False, mem=True, rl=True, branches=("text", "speech", "gemaps")),
    dict(name="5. speech + memory", video=False, mem=True, rl=False, branches=("text",)),
    dict(name="6. speech + vision", video=True, mem=False, rl=False, branches=("text",)),
    dict(name="7. speech + memory + vision", video=True, mem=True, rl=False, branches=("text",)),
    dict(name="8. speech + memory + vision + bandit", video=True, mem=True, rl=True, branches=("text",)),
]
ALL_BRANCHES = ("text", "speech", "gemaps")


def load_audio_rows(path: Path) -> list[dict]:
    """Records from data/preprocess_audio_dataset.py -> eval rows with speech branches."""
    import numpy as np
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("confirmed_meaning"):
            continue  # nothing to score against
        feats = np.load(ROOT / rec["speech_embedding_path"]) if rec.get("speech_embedding_path") else None
        cands = rec.get("intent_candidates") or []
        rows.append({
            "user_id": rec["speaker_id"], "session_id": rec["speaker_id"], "turn_index": len(rows),
            "fragment": rec.get("fragmented_utterance") or rec["transcript"],
            "pause_intervals": ";".join(str(x) for x in rec["pause_features"].get("pause_intervals", [])),
            "visible_objects": "", "pointing_target": "", "split": rec.get("split", "test"),
            "confirmed_text": rec["confirmed_meaning"], "candidate_1": cands[0] if cands else "",
            "_branches": {k: feats[k] for k in ALL_BRANCHES} if feats is not None else None,
            "_timing": feats["timing"] if feats is not None else None,
            "_pause": rec["pause_features"],
        })
    return rows


def speech_features_for(row: dict, branches: tuple) -> dict | None:
    """Rebuild the fused vector from the chosen branches (the ablation switch)."""
    if not row.get("_branches"):
        return None
    from backend.speech import fusion
    from backend.speech.schemas import SpeechObservations, VadStats
    b = row["_branches"]
    pick = lambda k: b[k] if (k in branches and b[k].size) else None  # noqa: E731
    fused = fusion.fuse(pick("text"), pick("speech"), pick("gemaps"), row["_timing"])
    pf = row["_pause"]
    obs = SpeechObservations(transcript=row["fragment"], vad=VadStats(**{k: v for k, v in pf.items() if k in VadStats.model_fields}),
                             fragmented=True, providers={"asr": "dataset"})
    return {"observations": obs.model_dump(), "fused_embedding": [float(x) for x in fused]}


def load_rows(limit: int | None) -> list[dict]:
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["user_id"], int(r["turn_index"])))  # chronological per user
    return rows[:limit] if limit else rows


def run_config(rows: list[dict], cfg: dict) -> dict:
    con = memory.connect(":memory:")
    for u in {r["user_id"] for r in rows}:
        memory.set_profile(con, u, SupportProfile(
            support_mode="prefer_not_to_say", camera_enabled=cfg["video"]))

    stats = {"n": 0, "top1": 0, "top3": 0, "coverage": 0, "reward": 0,
             "turns": 0, "latency_ms": 0.0}
    curve: dict[int, list[int]] = {}      # turn bucket -> hits (all splits)
    failures: dict[str, int] = {}
    for r in rows:
        objs = [o for o in r["visible_objects"].split(";") if o] if cfg["video"] else []
        t0 = time.perf_counter()
        out = pipeline.process(con, ProcessReq(
            user_id=r["user_id"], session_id=r["session_id"], transcript=r["fragment"],
            pause_intervals=[float(x) for x in r["pause_intervals"].split(";") if x],
            scene_hint=objs,
            pointing_hint=(r["pointing_target"] or None) if cfg["video"] else None,
            speech_features=speech_features_for(r, cfg["branches"])))
        dt = (time.perf_counter() - t0) * 1000

        truth = r["confirmed_text"].strip().lower()
        cands = [c.strip().lower() for c in out["candidates"]]
        hit1 = bool(cands) and cands[0] == truth
        in_set = truth in cands

        pipeline.feedback(con, FeedbackReq(
            interaction_id=out["interaction_id"], accepted=hit1,
            confirmed_text=r["confirmed_text"], none_fit=not in_set,
            elapsed_ms=int(dt)), learn=cfg["rl"], remember=cfg["mem"])

        bucket = int(r["turn_index"]) // 10
        curve.setdefault(bucket, []).append(int(hit1))
        if not hit1:
            failures["MISSING_CANDIDATE" if not in_set else "RANKING_ERROR"] =                 failures.get("MISSING_CANDIDATE" if not in_set else "RANKING_ERROR", 0) + 1

        if r["split"] != "test":
            continue  # train/dev turns only shape the policy and memory
        stats["n"] += 1
        stats["top1"] += hit1
        stats["top3"] += in_set
        stats["coverage"] += in_set
        stats["reward"] += 1 if hit1 else -1
        stats["turns"] += 0 if hit1 else (1 if in_set else 2)
        stats["latency_ms"] += dt

    n = max(1, stats["n"])
    return {"name": cfg["name"], "n_test": stats["n"],
            "top1": stats["top1"] / n, "top3": stats["top3"] / n,
            "candidate_coverage": stats["coverage"] / n,
            "avg_reward": stats["reward"] / n,
            "clarification_burden": stats["turns"] / n,
            "latency_ms_per_interaction": stats["latency_ms"] / n,
            "curve": {b: sum(v) / len(v) for b, v in sorted(curve.items())},
            "failure_counts": failures}


def log_wandb(report: dict) -> None:
    """One W&B run per evaluation: ablation table + headline metrics."""
    if not os.getenv("WANDB_API_KEY"):
        return
    try:
        import wandb
        project = os.getenv("WANDB_PROJECT", "echoloop").split("/")[-1]  # bare name
        run = wandb.init(entity=os.getenv("WANDB_ENTITY") or None, project=project,
                         job_type="evaluation", config={"rows": report["rows_replayed"]})
        table = wandb.Table(columns=["configuration", "top1", "top3", "candidate_coverage",
                                     "avg_reward", "clarification_burden", "latency_ms"])
        for a in report["ablations"]:
            table.add_data(a["name"], a["top1"], a["top3"], a["candidate_coverage"],
                           a["avg_reward"], a["clarification_burden"],
                           a["latency_ms_per_interaction"])
        best = report["ablations"][-1]
        run.log({"ablations": table,
                 "top1_accuracy": best["top1"], "top3_accuracy": best["top3"],
                 "avg_reward": best["avg_reward"],
                 "clarification_burden": best["clarification_burden"],
                 "candidate_coverage": best["candidate_coverage"],
                 "memory_hit_rate": report["memory_hit_rate"],
                 "personalization_gain": report["personalization_gain"],
                 "time_to_confirmed_ms": best["latency_ms_per_interaction"]})
        run.finish()
        print("logged evaluation run to W&B")
    except Exception as e:
        print(f"[wandb] evaluation run not logged: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke runs)")
    ap.add_argument("--audio-dataset", type=Path, default=None,
                    help="JSONL from data/preprocess_audio_dataset.py (enables speech ablations)")
    args = ap.parse_args()

    weave_init()
    if args.audio_dataset:
        rows = load_audio_rows(args.audio_dataset)
        rows.sort(key=lambda r: (r["user_id"], r["turn_index"]))
        rows = rows[: args.limit] if args.limit else rows
    else:
        rows = load_rows(args.limit)
    results = [run_config(rows, cfg) for cfg in ABLATIONS]
    frozen = next(r for r in results if r["name"].startswith("7"))
    learned = next(r for r in results if r["name"].startswith("8"))
    has_audio = bool(rows and rows[0].get("_branches"))
    for r in results:
        r["speech_features_available"] = has_audio
        if not has_audio and r["name"][0] in "234":
            r["note"] = "no audio features in this dataset: identical to transcript-only + memory + bandit"

    report = {
        "dataset": (f"audio dataset {args.audio_dataset.name}" if args.audio_dataset else
                    "EchoLoop 5,000-example synthetic interaction environment "
                    "(staged engineering data, not real user or clinical data)"),
        "rows_replayed": len(rows), "n_test": learned["n_test"],
        "ablations": results,
        "personalization_gain": learned["top1"] - frozen["top1"],
        "memory_hit_rate": learned["candidate_coverage"],
        "compute": {"device": os.getenv("COREWEAVE_ENDPOINT", "local CPU"),
                    "examples_processed": len(rows) * len(ABLATIONS),
                    "models": "hashed-embedding retrieval + linear contextual bandit"},
        "integrations": status(),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (ROOT / "data" / "eval_curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "turn_bucket", "top1_accuracy"])
        for r in results:
            for b, acc in r["curve"].items():
                w.writerow([r["name"], b * 10, round(acc, 4)])
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_trace("evaluation.report", report, force=True)
    log_wandb(report)

    print(f"\n{'configuration':44} {'top1':>7} {'top3':>7} {'reward':>8} {'turns':>7} {'ms':>7}")
    for r in results:
        print(f"{r['name']:44} {r['top1']:7.3f} {r['top3']:7.3f} {r['avg_reward']:8.3f} "
              f"{r['clarification_burden']:7.2f} {r['latency_ms_per_interaction']:7.2f}")
    print(f"\npersonalization gain (8 vs 7): {report['personalization_gain']*100:+.1f} pts top-1")
    if not has_audio:
        print("note: ablations 2-4 need --audio-dataset (GeMAPS/WavLM features); "
              "on the synthetic CSV they equal transcript + memory + bandit")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
