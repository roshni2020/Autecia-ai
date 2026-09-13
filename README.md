# EchoLoop

**A live multimodal communication assistant that uses four cooperating agents and explicit
reinforcement feedback to learn how one individual expresses incomplete thoughts.**

> You do not learn how to talk to the AI. The AI learns how you communicate.

EchoLoop listens to an incomplete utterance ("I need… that… blue…"), optionally looks at what
is visibly in the room, proposes a few complete sentences, and **only speaks what the user
confirms**. Every confirmation or correction is a reward signal that reranks the next
similar interaction for that person.

## What it does not do

- No autism detection, no diagnosis, no clinical claim.
- No emotion, mood, anxiety, or sensory-state inference — from faces or anything else.
- Support mode is **chosen by the user in onboarding** and is a preference, never inferred.
- Nothing is spoken aloud unless the user confirmed that exact text.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # optional keys; everything runs without them
python -m data.generate_synthetic     # writes the 5,000-row CSV (already committed)
uvicorn backend.main:app --reload     # open http://127.0.0.1:8000
```

Checks and evaluation:

```bash
python -m tests.test_loop     # 6 assertions over the learning loop
python -m eval.run_eval       # 5 ablations over 5,000 interactions -> W&B + report
marimo edit notebooks/evaluation_marimo.py    # molab notebook: curves + charts
```

## Two-interaction demo

1. Onboard, pick a support mode, turn the camera on (or leave the scene box as
   `blue headphones, blue notebook`).
2. Say or type **"I need… blue…"** → three candidates. Top one is *"I need my notebook."*
3. Press **Not quite**, correct it to **"I need my headphones."**
   → `Reward: -1`, correction stored, memory updated, policy version bumped,
   Reflection: `RANKING_ERROR / VISUAL_AND_MEMORY_UNDERWEIGHTED`.
4. Say **"Can you get… blue thing…"** → *"I need my headphones."* is now ranked **#1**.
5. Press **Yes** → `Reward: +1` → **Speak this** says the confirmed sentence.

That before/after ranking flip is asserted in `tests/test_loop.py::test_correction_changes_ranking`,
so it cannot silently rot.

## The four agents

Each lives in its own folder under [backend/agents/](backend/agents/) and exchanges typed
JSON messages (`message_type / from / to / payload`), assembled in [backend/pipeline.py](backend/pipeline.py).

| Agent | Folder | Does |
| --- | --- | --- |
| Perception | [backend/agents/perception/](backend/agents/perception/) | transcript, pauses, repetition, visible objects, pointing. Observable facts only. |
| Intent | [backend/agents/intent/](backend/agents/intent/) | 2–4 diverse candidate sentences + a "None of these" path. |
| Learning | [backend/agents/learning/](backend/agents/learning/) | retrieves confirmed memories, builds features, reranks, updates the policy. |
| Reflection | [backend/agents/reflection/](backend/agents/reflection/) | labels why a turn succeeded or failed (`RANKING_ERROR`, `MISSING_CANDIDATE`, …). Recommends only — never overrides the user. |

## The learning loop

Features per candidate: `base, memory, visual, history, pointing, brevity, bias`.
Score is `w · features`; weights are per user, seeded by support mode, persisted in SQLite.

```
accepted top suggestion   -> reward +1
rejected top suggestion   -> reward -1
user picked/edited another -> that text becomes the confirmed label (supervised preference step)
none of these             -> no ranking update (the generator missed, not the ranker)
```

Two details that matter, both found by evaluation rather than assumed:

- **Advantage baseline.** Ranking depends only on *differences* between candidates, so the
  update uses `features − mean(candidate features)`. Updating on raw features moved every
  candidate together and made the RL layer *worse* than a frozen policy (−17.6 pts).
- **Generator misses are not ranker errors.** When the confirmed sentence was never generated,
  the ranking weights are left alone.

## Results (5,000-row synthetic environment, held-out test split)

| Configuration | Top-1 | Top-3 | Avg reward | Clarification turns |
| --- | --- | --- | --- | --- |
| 1. speech only | 3.2% | 3.2% | −0.94 | 1.94 |
| 2. speech + video | 21.2% | 26.3% | −0.58 | 1.52 |
| 3. speech + memory | 21.9% | 33.5% | −0.56 | 1.45 |
| 4. speech + video + memory | 32.4% | 47.6% | −0.35 | 1.20 |
| 5. + contextual-bandit reranking | **36.7%** | 47.6% | **−0.27** | **1.16** |

**Personalization gain: +4.3 points top-1** from the RL layer over the identical system with a
frozen policy. Replay is chronological per user — no later feedback reaches an earlier
prediction. Regenerate with `python -m eval.run_eval`.

## Data

`data/EchoLoop_5000_Synthetic_Interactions.csv` — 5,000 rows × 25 columns, 40 simulated users
with stable personal preferences, split 3,520 / 760 / 720 chronologically per user.

> This is a **synthetic interaction environment used to bootstrap and evaluate the adaptive
> ranking loop**. It is staged engineering data. It is not real user data, not autistic-user
> data, and not clinical data.

Generated by [data/generate_synthetic.py](data/generate_synthetic.py) — drop in your own CSV at
the same path with the same columns and everything reads it instead.

Other sources are kept **separate by purpose**, never joined into one fake multimodal corpus:

| Source | Purpose | Status |
| --- | --- | --- |
| AAC-like corpus (aactext.org/imagine) | phrase bank / candidate language | paste sentences into `data/phrasebank.txt` |
| COMM2 (aactext.org/comm2) | external text evaluation | same loader |
| YouRefIt | pointing + referent grounding benchmark | not wired; perception eval only |
| ASDBank AAC, HeyJay!, SEP-28k | research grounding, atypical speech, disfluency stress | controlled access; demo does not depend on them |

## Integrations

Everything degrades to a working offline path, so the demo never depends on a network call.

| Service | Used for | Without a key |
| --- | --- | --- |
| **W&B Weave** | traces every agent call, interaction, reward, reflection | JSONL at `data/traces.jsonl` |
| **W&B runs** | one run per evaluation: ablation table + headline metrics | report JSON only |
| **ElevenLabs** | speaking confirmed text | browser `speechSynthesis` |
| **Gemini** | frame → visible objects + pointing (observable facts only) | scene box in the UI |
| **TypeSafe AI** (System One, `jev`) | per-candidate "which one is meant?" probability → a bandit feature; failure-type classifier in Reflection | feature is 0; heuristic reflection |
| **CoreWeave** | batch evaluation / inference host | local CPU, recorded in the report |

Keys go in `.env` (git-ignored). Batch jobs set `ECHOLOOP_TRACE=0` — a span per agent call turns
a 5,000-row replay into an hours-long job.

W&B MCP server, if you want the dashboards inside Claude Code:

```bash
claude mcp add --transport http wandb https://mcp.withwandb.com/mcp \
  --header "Authorization: Bearer $WANDB_API_KEY"
```

## TypeSafe System One

Two judgments per interaction, both via `client.system_one()` ([backend/integrations.py](backend/integrations.py) `typesafe_judge`):

- **Learning agent** — a `Choice` over the candidate sentences (+ "none of these") given the utterance,
  visible objects, pointing target and this person's confirmed history. The returned probability becomes
  the `judgment` feature. It is evidence, not the decision: the per-user bandit learns how much to trust it
  next to memory and vision, so explicit user feedback still wins.
- **Reflection agent** — a `Choice` over the nine failure types with plain-language criteria, replacing the
  rule-based classifier whenever TypeSafe is reachable.

Batch replays run with `ECHOLOOP_TYPESAFE=0` so an evaluation is not 25,000 network calls.

## Voice and camera

- **Input**: browser speech recognition, continuous, with pause tolerance from the support mode
  (long = 6s of silence before it assumes you are finished). The transcript is passed through
  **unedited** — fillers, repeats and fragments are the signal, not noise to clean up.
- **Output**: ElevenLabs or the browser voice, confirmed text only, with a **Stop** button.
- **Camera**: off by default, one click to disable. Detection runs **in the browser**:
  TensorFlow.js COCO-SSD boxes objects (the `person` class is discarded — never drawn, never sent) and
  MediaPipe Hands turns an extended index finger into a ray; the object it hits is the pointing target.
  Detected labels and the pointing target go to the Perception agent as scene context. Frames stay
  on-device unless a server vision key (`GEMINI_API_KEY`) is configured. Nothing is stored.
  COCO-SSD knows 80 everyday classes (book, cup, bottle, cell phone, laptop, remote, scissors, …) —
  not "headphones", so demo with a book and a cup.

## User control

"None of these", free-text edit, camera off, stop speaking, and **Forget this memory** on every
stored item (Learning History tab). Raw audio and video are never persisted — only derived
context, embeddings, confirmed sentences and reward data.

## Layout

```
backend/     agents/{perception,intent,learning,reflection}/  pipeline.py  bandit.py
             memory.py  embed.py  integrations.py  schemas.py  main.py
frontend/    index.html            # one file: onboarding, live, history, trace, evaluation
data/        generate_synthetic.py  EchoLoop_5000_Synthetic_Interactions.csv  phrasebank.txt
eval/        run_eval.py           # ablations, curves, W&B run
notebooks/   evaluation_marimo.py  # molab
tests/       test_loop.py
```

## Deliberate shortcuts

Marked in code with `ponytail:` comments.

- Hashed bag-of-ngrams embeddings instead of a sentence transformer — offline, deterministic,
  no model download. Swap in a real embedder if retrieval quality becomes the bottleneck.
- NumPy cosine over one user's rows instead of a vector service. Move to Chroma/pgvector past
  ~10k memories per user.
- One static HTML page instead of a React/Vite build — no npm, no build step, same UI.
