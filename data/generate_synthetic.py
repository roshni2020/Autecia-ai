"""Generate EchoLoop_5000_Synthetic_Interactions.csv (spec §10).

SYNTHETIC / STAGED ENGINEERING DATA. Not real users, not clinical data,
not autism data. It exists to bootstrap and evaluate the ranking loop.

Usage: python -m data.generate_synthetic
"""
import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "EchoLoop_5000_Synthetic_Interactions.csv"
N_ROWS, N_USERS, SEED = 5000, 40, 7

# Each scenario: carrier fragment shapes -> the object/action the user meant.
REFERENTS = [
    ("headphones", "I need my headphones.", ["blue", "thing"]),
    ("notebook", "I need my notebook.", ["blue", "book"]),
    ("water", "I need some water.", ["drink", "cup"]),
    ("phone", "I need my phone.", ["black", "thing"]),
    ("jacket", "I need my jacket.", ["warm", "coat"]),
    ("charger", "I need my charger.", ["cable", "cord"]),
    ("glasses", "I need my glasses.", ["see", "read"]),
    ("keys", "I need my keys.", ["door", "metal"]),
    ("blanket", "I need my blanket.", ["cold", "soft"]),
    ("tablet", "I need my tablet.", ["screen", "app"]),
]
ACTIONS = [
    ("break", "I need a break.", ["stop", "tired", "much"]),
    ("quiet", "It is too loud.", ["loud", "noise", "much"]),
    ("leave", "I want to leave.", ["go", "out", "home"]),
    ("help", "I need help.", ["stuck", "cannot"]),
    ("alone", "I want to be alone for a bit.", ["people", "crowd"]),
]
FRAGMENT_SHAPES = [
    "I need... {cue}...", "can you get... {cue}...", "I... um... {cue}...",
    "{cue}...", "I want... that... {cue}...", "can we... {cue}...",
    "I need... that... {cue}...", "uh... {cue}... {cue}...",
]
SUPPORT_MODES = ["autism_neurodivergent", "speech_language", "aphasia_word_finding",
                 "cognitive_fatigue", "aac", "other", "prefer_not_to_say"]

COLUMNS = ["interaction_id", "user_id", "session_id", "turn_index", "timestamp",
           "support_mode", "fragment", "fragment_word_count", "pause_intervals",
           "repetition_detected", "camera_enabled", "visible_objects", "pointing_target",
           "pointing_confidence", "candidate_1", "candidate_2", "candidate_3",
           "top1_before_learning", "confirmed_text", "user_feedback", "reward",
           "memory_similarity", "learned_rank_score", "policy_version", "split"]


def main() -> None:
    rng = random.Random(SEED)
    users = [f"user_{i:02d}" for i in range(1, N_USERS + 1)]
    # every user has a stable personal preference: what they usually mean by a cue
    prefs = {u: rng.sample(REFERENTS, 3) for u in users}
    rows, t = [], 1_700_000_000

    for i in range(N_ROWS):
        u = users[i % N_USERS]
        turn = i // N_USERS
        target_pool = prefs[u] if rng.random() < 0.7 else REFERENTS
        target = rng.choice(target_pool + [(a, b, c) for a, b, c in ACTIONS])
        label, confirmed, cues = target
        cue = rng.choice(cues)
        fragment = rng.choice(FRAGMENT_SHAPES).format(cue=cue)

        distractors = [r for r in REFERENTS if r[0] != label]
        d1, d2 = rng.sample(distractors, 2)
        camera = rng.random() < 0.6
        visible = [label, d1[0]] if camera else []
        if camera:
            rng.shuffle(visible)
        pointing = label if (camera and rng.random() < 0.45) else ""
        point_conf = round(rng.uniform(0.55, 0.95), 2) if pointing else 0.0

        cands = [confirmed, d1[1], d2[1]]
        rng.shuffle(cands)
        # before learning: pointing dominates, otherwise near-random among candidates
        if pointing and rng.random() < 0.8:
            top1 = confirmed
        else:
            top1 = cands[0] if rng.random() < 0.45 else rng.choice(cands)

        # memory similarity grows as the same user repeats a preference
        seen = sum(1 for r in rows if r["user_id"] == u and r["confirmed_text"] == confirmed)
        mem_sim = round(min(0.95, 0.1 + 0.18 * seen + rng.uniform(-0.05, 0.05)), 3)

        accepted = top1 == confirmed
        rows.append({
            "interaction_id": f"i_{i:05d}", "user_id": u,
            "session_id": f"{u}_s{turn // 8:02d}", "turn_index": turn,
            "timestamp": t + i * 37, "support_mode": SUPPORT_MODES[hash(u) % len(SUPPORT_MODES)],
            "fragment": fragment,
            "fragment_word_count": len(fragment.replace("...", " ").split()),
            "pause_intervals": ";".join(str(round(rng.uniform(0.6, 2.8), 1))
                                        for _ in range(rng.randint(1, 2))),
            "repetition_detected": int("{cue}... {cue}" in fragment or rng.random() < 0.12),
            "camera_enabled": int(camera), "visible_objects": ";".join(visible),
            "pointing_target": pointing, "pointing_confidence": point_conf,
            "candidate_1": cands[0], "candidate_2": cands[1], "candidate_3": cands[2],
            "top1_before_learning": top1, "confirmed_text": confirmed,
            "user_feedback": "accepted" if accepted else "rejected",
            "reward": 1 if accepted else -1, "memory_similarity": mem_sim,
            "learned_rank_score": round(0.4 + 0.3 * mem_sim + 0.3 * point_conf, 3),
            "policy_version": 1 + seen, "split": "",
        })

    # chronological split per user: no later feedback leaks into earlier predictions
    for u in users:
        idx = [j for j, r in enumerate(rows) if r["user_id"] == u]
        n = len(idx)
        for k, j in enumerate(idx):
            rows[j]["split"] = ("train" if k < 0.7 * n else "dev" if k < 0.85 * n else "test")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "dev", "test")}
    print(f"wrote {OUT} rows={len(rows)} cols={len(COLUMNS)} splits={counts}")


if __name__ == "__main__":
    main()
