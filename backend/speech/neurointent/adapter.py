"""Adapter over the vendored NeuroIntent inference code (see NOTICE.md).

Uses, unmodified: `_to_wav`, `_load_models` (Whisper base / openSMILE GeMAPSv01b /
RoBERTa-large / trained FusionLayer) and `FusionLayer`. The hiring-specific parts
of `run()` (intent classes, scores, interpretations) are not called.

Checkpoint: their loader looks for  backend/models/checkpoints/fusion_layer_trained_v1.pt
(download it from the upstream repo). Without it the same models are loaded with
fusion=None and the content/prosody representation falls back to the RoBERTa CLS.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import pipeline as ni

ENABLED = os.getenv("ECHOLOOP_NEUROINTENT", "1") != "0"


@lru_cache(maxsize=1)
def load_models() -> dict | None:
    if not ENABLED:
        return None
    try:  # torch + CTranslate2 both spawn OpenMP pools; capped they stop fighting for cores
        import torch
        torch.set_num_threads(int(os.getenv("ECHOLOOP_TORCH_THREADS", "4")))
    except Exception:
        pass
    try:
        return ni._load_models()                      # their loader, checkpoint included
    except FileNotFoundError as e:
        print(f"[neurointent] no fusion checkpoint, loading models without it: {e}")
    except Exception as e:
        print(f"[neurointent] unavailable: {e}")
        return None
    try:                                              # same models as their loader, minus fusion
        import opensmile
        import torch
        from faster_whisper import WhisperModel
        from transformers import RobertaModel, RobertaTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Their loader fixes roberta-large (1.4 GB). Without the checkpoint the CLS size
        # is free, so a smaller RoBERTa can stand in (NEUROINTENT_TEXT_MODEL=roberta-base).
        text_model = os.getenv("NEUROINTENT_TEXT_MODEL", "roberta-large")
        return {"fusion": None,
                "smile": opensmile.Smile(feature_set=opensmile.FeatureSet.GeMAPSv01b,
                                         feature_level=opensmile.FeatureLevel.Functionals),
                "whisper": WhisperModel(os.getenv("WHISPER_MODEL", "base"), device=device,
                                        compute_type="float16" if device == "cuda" else "int8"),
                "tokenizer": RobertaTokenizer.from_pretrained(text_model),
                "roberta": RobertaModel.from_pretrained(text_model).eval(),
                "device": device}
    except Exception as e:
        print(f"[neurointent] models unavailable: {e}")
        return None


def available() -> bool:
    return load_models() is not None


def to_wav(audio_path: str) -> tuple[str, bool]:
    """Their ffmpeg conversion. Returns (wav_path, needs_cleanup)."""
    return ni._to_wav(audio_path)


def analyze(wav_path: Path, transcript_override: str | None = None) -> dict | None:
    """GeMAPS + Whisper(+word timestamps) + RoBERTa CLS + fusion representation.

    Mirrors steps 1-4 of their `run()`; returns raw features instead of hiring
    scores. `transcript_override` lets datasets with verified transcripts skip ASR.
    """
    m = load_models()
    if m is None:
        return None
    import torch
    device = m["device"]
    out: dict = {}
    # 1. GeMAPS (their extraction, 62 functionals)
    prosody_df = m["smile"].process_file(str(wav_path))
    prosody_np = prosody_df.values[0].astype(np.float32)
    out["gemaps"], out["feature_names"] = prosody_np, list(prosody_df.columns)
    # 2. Whisper (their call + word timestamps + disfluency prompt: wording is kept raw)
    if transcript_override is None:
        segments, info = m["whisper"].transcribe(
            str(wav_path), beam_size=int(os.getenv("WHISPER_BEAM", "1")), word_timestamps=True, condition_on_previous_text=False,
            initial_prompt="Um, uh... I, I need... that, um, the... you know... hmm.")
        words, segs, texts = [], [], []
        for s in segments:
            segs.append({"start": round(s.start, 3), "end": round(s.end, 3)})
            texts.append(s.text.strip())
            for w in (s.words or []):
                words.append({"word": w.word.strip(), "start": round(w.start, 3),
                              "end": round(w.end, 3), "probability": round(w.probability, 3)})
        out["transcript"] = " ".join(t for t in texts if t).strip()
        out["words"], out["segments"] = words, segs
        out["language"], out["language_probability"] = info.language, round(info.language_probability, 3)
        out["asr_confidence"] = round(sum(w["probability"] for w in words) / len(words), 3) if words else 0.0
        out["asr_model"] = "base"
    else:
        out.update(transcript=transcript_override, words=[], segments=[], language=None,
                   language_probability=0.0, asr_confidence=0.0, asr_model=None)
    # 3. RoBERTa-large CLS (their lines)
    inputs = m["tokenizer"](out["transcript"] or "[no speech detected]", return_tensors="pt",
                            max_length=512, truncation=True)
    with torch.no_grad():
        cls_emb = m["roberta"](**inputs).last_hidden_state[:, 0, :]        # [1, 1024]
    out["cls"] = cls_emb.squeeze(0).cpu().numpy().astype(np.float32)
    out["text_model"] = getattr(getattr(m["roberta"], "config", None), "_name_or_path", "roberta-large")
    # 4. FusionLayer content/prosody representation (their forward; logits ignored)
    if m["fusion"] is not None:
        prosody_t = torch.tensor(prosody_np).unsqueeze(0).to(device)
        with torch.no_grad():
            C, D, _logits = m["fusion"](cls_emb.to(device), prosody_t)
            rep = torch.cat([C, D, C - D, C * D], dim=-1).squeeze(0)
            out["fusion_vec"] = rep.cpu().numpy().astype(np.float32)          # [1024]
            out["cos_cd"] = float(torch.nn.functional.cosine_similarity(C, D, dim=-1).item())
    else:
        out["fusion_vec"], out["cos_cd"] = None, None
    return out
