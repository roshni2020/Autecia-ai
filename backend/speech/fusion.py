"""Multimodal fusion -> one utterance representation for intent recovery.

Branches: text (transcript), speech (WavLM), prosody (GeMAPS), timing (VAD).
Two paths:

* `fuse()`   — deterministic, always available: each branch L2-normalised and
               concatenated with fixed weights. This is what the live pipeline
               uses today. It needs no training and degrades branch-by-branch.
* `FusionModel` — a small torch module: a content branch
               (text+speech) and a delivery branch (prosody+timing) projected to
               256-d, combined as [C, D, C-D, C*D] -> 256-d utterance vector.
               Loaded only if models/fusion.pt exists; trained by
               data/preprocess_audio_dataset.py output + confirmed meanings.
               It has NO intent-class head: Autecia never scores how someone
               speaks, it recovers what they meant.
"""
import os
from pathlib import Path

import numpy as np

CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "fusion.pt"
TIMING_DIM = 6
BRANCH_WEIGHTS = {"text": 1.0, "speech": 0.8, "prosody": 0.5, "timing": 0.3}


def _unit(v: np.ndarray | None, dim: int) -> np.ndarray:
    if v is None:
        return np.zeros(dim, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def timing_vector(vad: dict, fragmented: bool, repetition: bool, speaking_rate_wpm: float) -> np.ndarray:
    return np.array([
        min(1.0, vad.get("speech_duration", 0.0) / 10.0),
        min(1.0, vad.get("pause_count", 0) / 6.0),
        min(1.0, vad.get("longest_pause_ms", 0) / 3000.0),
        1.0 if fragmented else 0.0,
        1.0 if repetition else 0.0,
        min(1.0, speaking_rate_wpm / 200.0),
    ], dtype=np.float32)


def scale_gemaps(g: np.ndarray | None) -> np.ndarray | None:
    """Signed log compression: GeMAPS functionals span many orders of magnitude and
    we have no neurotypical-baseline statistics (and do not want to normalise a
    person against one). Per-utterance scaling only."""
    if g is None:
        return None
    g = np.asarray(g, dtype=np.float32)
    return np.sign(g) * np.log1p(np.abs(g))


def fuse(text_emb: np.ndarray | None, speech_emb: np.ndarray | None,
         gemaps: np.ndarray | None, timing: np.ndarray | None) -> np.ndarray:
    """Deterministic fusion. Output length depends on which branches exist, so
    similarity is only computed between vectors of equal length (see learning)."""
    parts = [
        BRANCH_WEIGHTS["text"] * _unit(text_emb, 0) if text_emb is not None else np.zeros(0, np.float32),
        BRANCH_WEIGHTS["speech"] * _unit(speech_emb, 0) if speech_emb is not None else np.zeros(0, np.float32),
        BRANCH_WEIGHTS["prosody"] * _unit(scale_gemaps(gemaps), 0) if gemaps is not None else np.zeros(0, np.float32),
        BRANCH_WEIGHTS["timing"] * _unit(timing, TIMING_DIM) if timing is not None else np.zeros(TIMING_DIM, np.float32),
    ]
    v = np.concatenate(parts).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape or not a.size:
        return 0.0
    d = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
    return max(0.0, min(1.0, d))


# ---- learnable fusion (optional; content/delivery interaction) -----------------

def build_model(text_dim: int = 768, speech_dim: int = 768, gemaps_dim: int = 62,
                timing_dim: int = TIMING_DIM, hidden: int = 256):
    import torch
    from torch import nn

    class Branch(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden))

        def forward(self, x):
            return self.net(x)

    class FusionModel(nn.Module):
        """content C = f(text, speech); delivery D = g(gemaps, timing);
        utterance = h([C, D, C-D, C*D]). No classification head."""
        def __init__(self):
            super().__init__()
            self.content = Branch(text_dim + speech_dim)
            self.delivery = Branch(gemaps_dim + timing_dim)
            self.out = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

        def forward(self, text, speech, gemaps, timing):
            c = self.content(torch.cat([text, speech], dim=-1))
            d = self.delivery(torch.cat([gemaps, timing], dim=-1))
            return torch.nn.functional.normalize(self.out(torch.cat([c, d, c - d, c * d], dim=-1)), dim=-1)

    return FusionModel()


def load_fusion_model():
    """Trained fusion model if a checkpoint exists; otherwise None (use fuse())."""
    if not CHECKPOINT.exists() or os.getenv("ECHOLOOP_FUSION", "1") == "0":
        return None
    try:
        import torch
        state = torch.load(CHECKPOINT, map_location="cpu")
        model = build_model(**state.get("dims", {}))
        model.load_state_dict(state["state_dict"]); model.eval()
        return model
    except Exception as e:
        print(f"[fusion] checkpoint unusable, deterministic fusion: {e}")
        return None
