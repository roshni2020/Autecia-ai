# EchoLoop

**A multi-agent communication assistant that learns how one person expresses incomplete thoughts — then speaks and sends what they confirm.**

> You do not learn how to talk to the AI. The AI learns how you communicate.

Built by **Roshni Kobula** (roshnikobula2020@gmail.com), **Ali Amjad** (ali.amjad52114@gmail.com) and **Rikin Shah** (rshah88@asu.edu) for the Multi-App AI Agent Hackathon (Lemma × Comma Capital, Sept 13 2026).

- Repository: https://github.com/roshni2020/Autecia-ai
- **Live app:** https://auteciia.vercel.app (hosted light mode: browser speech recognition, no Slack; full stack runs locally per §03)
- Demo video (1:23): https://github.com/roshni2020/Autecia-ai/blob/main/docs/EchoLoop_demo.mp4
- Technical deep-dive: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 01 · Project overview

Many autistic and neurodivergent people, people with aphasia or speech difficulty, and AAC users know exactly what they want to say but cannot always finish or organise the sentence. *"I need… that… blue…"* — and everyone around them starts guessing. Every fragment becomes a clarification loop.

EchoLoop listens to the fragment exactly as spoken, looks at what is visibly in the room (objects, pointing), remembers what this person has confirmed before, and proposes 2–4 complete sentences. The person picks one — or says none fit. Only then does EchoLoop **speak it aloud** and, if they choose, **send it to their caregiver's Slack channel**. Every confirmation or correction is a reward that changes how the next similar fragment is ranked *for that person*.

**One multi-step agent, four cooperating roles, five external apps, one loop per utterance:**

```
mic / camera ─► Perception agent ─► Intent agent ─► Learning agent ─► user confirms ─► speak  (ElevenLabs)
                Whisper, VAD,       W&B Inference    TypeSafe judge +                 ─► send   (Slack, via Arga twin)
                GeMAPS, objects,    Qwen3-14B        per-user bandit,                 ─► reward → memory + policy
                pointing                             memory retrieval                 ─► Reflection agent
                                                          every step traced in W&B Weave (Agents dashboard, Evals)
```

What it deliberately is **not**: an autism detector, an emotion detector, or a clinical tool. Support mode is *chosen* by the user in onboarding, never inferred; nothing is inferred from faces; nothing is spoken or sent unless the user confirmed that exact text.

## 02 · External apps used

| App | What the agent does with it | Where |
| --- | --- | --- |
| **W&B Inference** (CoreWeave-hosted `OpenPipe/Qwen3-14B-Instruct`) | Intent agent generates grounded candidate sentences from the fragment, visible objects, pointing and the person's confirmed history | `backend/integrations.py` `wandb_inference` |
| **TypeSafe AI** (System One, `jev`) | Learning agent asks *"which candidate does this person mean?"* → calibrated probability per candidate, used as a bandit feature; Reflection agent classifies why a miss happened (9 failure types) | `backend/agents/learning`, `backend/agents/reflection` |
| **ElevenLabs** | Speaks the confirmed message (and Echo's *"Do you mean…?"*) with word timestamps that drive the 3D avatar's lip-sync | `backend/integrations.py` `tts` |
| **Slack** — tested on an **Arga Labs** twin | *Send to caregiver*: posts the confirmed message to `#echoloop-messages` via the real Slack Web API. In development and tests the same code talks to an Arga Labs Slack twin (a seeded, isolated API twin), so no real workspace is touched | `backend/integrations_slack.py`, `eval/arga_twin.py` |
| **W&B Weave** | Every agent step is a traced op; the four agents appear in the Weave **Agents** dashboard with LLM and TypeSafe tool spans; `weave.Evaluation` runs in the Evals tab | `backend/agent_trace.py`, `eval/weave_eval.py` |

Also used: Ready Player Me / TalkingHead (3D avatar), TensorFlow.js COCO-SSD + MediaPipe Hands (in-browser object and pointing detection), faster-whisper, Silero VAD, openSMILE, W&B Runs.

**How Arga Labs is used.** `python -m eval.arga_twin` calls Arga's MCP endpoint (`create_twin_run`) to provision a Slack twin seeded from a scenario prompt — a caregiver circle with `#echoloop-messages`, `#general`, three members and prior messages — and writes its URL and bot token into `.env`. EchoLoop's Slack code is identical for twin and production (only `SLACK_BASE_URL` differs), so the outward action is exercised end to end without side effects. `tests/test_slack_twin.py` verifies that unconfirmed text is refused (HTTP 409) and confirmed text lands in the channel.

## 03 · Setup instructions

```bash
git clone https://github.com/roshni2020/Autecia-ai && cd Autecia-ai
pip install -r requirements.txt                 # core: FastAPI, agents, bandit, UI
pip install -r requirements-speech.txt          # optional: Whisper, VAD, GeMAPS, WavLM (~1.5 GB of models on first use)
cp .env.example .env                            # add keys (all optional; see below)
python -m uvicorn backend.main:app --port 8000  # open http://127.0.0.1:8000 in Chrome; allow mic + camera
```

Keys in `.env` — every one is optional; the app degrades gracefully without it:

| Key | Enables | Without it |
| --- | --- | --- |
| `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` | W&B Inference candidates, Weave tracing / Agents / Evals | template candidates, JSONL traces |
| `TYPESAFE_API_KEY` | System One judgments | feature = 0, heuristic reflection |
| `ELEVENLABS_API_KEY` | ElevenLabs voice + lip-sync | browser voice |
| `SLACK_BOT_TOKEN`, `SLACK_BASE_URL` | Send to caregiver | button disabled |
| `ARGA_API_KEY` | `python -m eval.arga_twin` provisions a Slack twin and fills the two Slack vars | use a real Slack bot token |

Demo script (~90 s): pick a support mode → camera on with a book and a cup in view (or *Add scene context manually*) → say **"I need… blue…"** → *Not quite* → type the real meaning → say **"Can you get… blue thing…"** → the corrected meaning is now #1 → *Yes* → *Speak with voice* → *Send to caregiver*. First utterance after start takes ~25 s while speech models load; afterwards ~5 s.

## 04 · Reliability testing

**Automated (all green on the submission commit):**

| Suite | Command | Covers |
| --- | --- | --- |
| Learning loop | `python -m tests.test_loop` | one correction flips the ranking; accepted = +1; none-fit is never speakable; camera-off yields no visual context; bandit advantage update; Forget removes memory |
| Speech layer | `python -m tests.test_speech` | WebM→16 kHz mono WAV with temp cleanup; VAD finds pauses; Whisper transcribes synthesized speech with word timestamps; GeMAPS = 62 features; encoder embedding finite; speech facts reach Perception unchanged; end-to-end with precomputed features |
| Outward action | `python -m tests.test_slack_twin` | against an Arga Labs Slack twin: unconfirmed text → 409; confirmed text lands in `#echoloop-messages` |
| Dashboard controls | `node --test tests/dashboard.test.cjs tests/dashboard-vision.test.cjs` | mic / camera / speech / session controls in a simulated browser |

**Offline evaluation** — `python -m eval.run_eval` replays a 5,000-interaction **synthetic** environment (40 simulated users, chronological per user, no feedback leakage) through the real pipeline with 8 ablations. Held-out test split:

| Configuration | Top-1 | Top-3 | Avg reward | Clarification turns |
| --- | --- | --- | --- | --- |
| speech transcript only | 3% | 3% | −0.94 | 1.94 |
| + vision | 21% | 26% | −0.58 | 1.52 |
| + memory + vision | 32% | 49% | −0.36 | 1.19 |
| **+ memory + vision + contextual bandit** | **35%** | 49% | **−0.31** | **1.16** |

The learning layer adds **+2.5–4.3 pts top-1** over the identical frozen system and reduces clarification turns. Two bugs were found *by* this harness and fixed: raw-feature bandit updates made RL worse than frozen (−17.6 pts) until an advantage baseline was used; generator misses were wrongly punishing the ranker. Results are logged as a W&B run; `python -m eval.weave_eval` publishes a `weave.Evaluation` (top-1, top-3, candidate count, latency) to the Weave Evals tab.

**Observability** — every live interaction is a Weave call tree (`echoloop.process` → `perception_agent.run` → `intent_agent.run` → `wandb_inference.chat` → `learning_agent.judge` (TypeSafe) → `learning_agent.rerank`; `echoloop.feedback` → `reflection_agent.run`), and the four agents appear as separate agents in the Weave Agents dashboard with per-step input/output messages, LLM token usage and tool spans.

**Verified in a real browser** (headless Chrome with fake devices): record → server Whisper → candidates → confirm → send; 3D avatar loads; in-browser object detection runs. The demo video was captured from this flow.

**Honest limits.** The evaluation data is synthetic (staged engineering data — not real users, not clinical). Per-row accuracy against exact-match ground truth is low by design; the claim is *personalisation gain*, not absolute accuracy. Slack is tested on a twin, not a production workspace. Whisper `base` on CPU adds ~2 s per utterance.

## 05 · Demo video

**▶ Watch:** https://github.com/roshni2020/Autecia-ai/blob/main/docs/EchoLoop_demo.mp4 (1:23, plays inline on GitHub) · [direct download](https://github.com/roshni2020/Autecia-ai/raw/main/docs/EchoLoop_demo.mp4)

Onboarding → fragment → wrong first guess → correction → same fragment ranks right → spoken by Echo → sent to the caregiver Slack twin → agent trace → evaluation → what's next (Lemma production monitoring, more Arga sandboxes, participatory study).
