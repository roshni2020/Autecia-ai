"""Normalise a speech dataset into EchoLoop training records.

    python -m data.preprocess_audio_dataset --manifest my_dataset.csv \
        --source my_corpus --license "CC BY-NC 4.0 (authorised copy)" \
        --out data/speech_records.jsonl

Manifest CSV columns (generic adapter):
    audio_path, speaker_id, [transcript], [confirmed_meaning], [fragmented_utterance],
    [intent_candidates (|-separated)], [reward], [consent]
Or a TalkBank/CHAT folder (--adapter chat --root DIR): *.cha transcripts with
matching audio files. Nothing is downloaded; you must already be authorised.

Each record keeps its provenance (source_dataset, license). Datasets are never
merged into one fake synchronised corpus: run this once per dataset and keep
the outputs separate. Splits are assigned per SPEAKER so a person never appears
in both train and test.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.speech import audio, fusion, observe, prosody, speech_encoder, text_encoder, vad, whisper  # noqa: E402
try:
    from backend.speech.neurointent import adapter as neurointent  # noqa: E402
except Exception:  # no torch: own modules only
    neurointent = None
from backend.speech.neurointent.utils import generate_speaker_id  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---------------- adapters: yield {"audio_path", "speaker_id", ...} -----------

def adapter_manifest(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield {k: (v or "").strip() for k, v in row.items()}


def adapter_chat(root: Path, speaker_codes=("CHI", "PAR", "SPE")):
    """TalkBank CHAT (.cha): one record per target-speaker utterance line.
    Expects a same-stem audio file next to each .cha (as TalkBank media exports).
    Utterance-level audio cuts need the %wor/bullet timings; when absent the whole
    file is used and `transcript` carries that utterance."""
    for cha in sorted(root.rglob("*.cha")):
        media = next((cha.with_suffix(ext) for ext in (".wav", ".mp3", ".mp4", ".m4a") if cha.with_suffix(ext).exists()), None)
        if media is None:
            continue
        speaker = cha.stem
        for line in cha.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^\*(\w+):\t(.*)$", line)
            if not m or m.group(1) not in speaker_codes:
                continue
            text = re.sub(r"\x15\d+_\d+\x15|\[.*?\]|&=?\w+|\(\.+\)|[<>]", " ", m.group(2))
            text = re.sub(r"\s+", " ", text).strip(" .")
            if text:
                yield {"audio_path": str(media), "speaker_id": speaker, "transcript": text,
                       "consent": "dataset-license"}


# ---------------- record building --------------------------------------------

def split_for(speaker_id: str) -> str:
    h = int(hashlib.sha1(speaker_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "dev" if h < 85 else "test"


def process_clip(row: dict, source: str, license_: str, feat_dir: Path, retranscribe: bool) -> dict | None:
    src = Path(row["audio_path"])
    if not src.exists():
        print(f"skip (missing audio): {src}")
        return None
    clip_id = hashlib.sha1(f"{source}:{src}:{row.get('transcript','')}".encode()).hexdigest()[:12]
    with audio.temp_wav(src.read_bytes(), src.suffix.lower() or ".wav") as wav:
        v = vad.analyze(wav)
        if v.total_duration < 0.5 or v.total_duration > 120:
            print(f"skip (duration {v.total_duration}s): {src}")
            return None
        ni = None
        if neurointent is not None and neurointent.available():
            ni = neurointent.analyze(wav, transcript_override=None if (retranscribe or not row.get("transcript"))
                                     else row["transcript"])
        if ni is not None:
            transcript, words, g = ni["transcript"], ni["words"], ni["gemaps"]
            tr = None
        else:
            tr = whisper.transcribe(wav) if (retranscribe or not row.get("transcript")) else None
            transcript = tr.text if tr else row.get("transcript", "")
            words = [w.model_dump() for w in tr.words] if tr else []
            g = prosody.extract(wav)
        s = speech_encoder.encode(wav)
        s_pooled = s[0] if s else None
    if ni is not None:
        t_emb = ni["fusion_vec"] if ni["fusion_vec"] is not None else ni["cls"]
        t_name = "roberta-large (neurointent)" + (" + FusionLayer v1" if ni["fusion_vec"] is not None else "")
    else:
        t_emb, t_name = text_encoder.encode(transcript)
    obs = observe(transcript, [], v, {})
    timing = fusion.timing_vector(v.model_dump(), obs.fragmented, obs.repetition_detected, obs.speaking_rate_wpm)

    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f"{clip_id}.npz"
    np.savez_compressed(feat_path, text=t_emb, speech=s_pooled if s_pooled is not None else np.zeros(0, np.float32),
                        gemaps=g if g is not None else np.zeros(0, np.float32), timing=timing)

    return {
        "interaction_id": clip_id,
        "speaker_id": row.get("speaker_id") or generate_speaker_id(),   # anonymised when absent
        "source_dataset": source, "audio_path": str(src),
        "transcript": transcript,
        "transcript_source": ("whisper:base (neurointent)" if (ni and ni.get("asr_model")) else
                              f"whisper:{tr.model}" if tr else "dataset"),
        "word_timestamps": words,
        "speech_embedding_path": str(feat_path.relative_to(ROOT)) if feat_path.is_relative_to(ROOT) else str(feat_path),
        "speech_encoder": speech_encoder.model_name() if s_pooled is not None else None,
        "text_encoder": t_name,
        "gemaps_features": [float(x) for x in g] if g is not None else [],
        "pause_features": v.model_dump(exclude={"speech_segments"}),
        "fragmented_utterance": row.get("fragmented_utterance") or transcript,
        "confirmed_meaning": row.get("confirmed_meaning", ""),
        "intent_candidates": [c for c in row.get("intent_candidates", "").split("|") if c],
        "reward": int(row["reward"]) if row.get("reward") not in (None, "") else None,
        "consent_or_dataset_license": row.get("consent") or license_,
        "split": split_for(row.get("speaker_id", "unknown")),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", choices=["manifest", "chat"], default="manifest")
    ap.add_argument("--manifest", type=Path, help="CSV manifest (manifest adapter)")
    ap.add_argument("--root", type=Path, help="folder of .cha + media (chat adapter)")
    ap.add_argument("--source", required=True, help="dataset name kept as provenance")
    ap.add_argument("--license", default="", help="license / authorisation note")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "speech_records.jsonl")
    ap.add_argument("--features-dir", type=Path, default=ROOT / "data" / "speech_features")
    ap.add_argument("--retranscribe", action="store_true", help="run Whisper even if a transcript exists")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    rows = adapter_manifest(a.manifest) if a.adapter == "manifest" else adapter_chat(a.root)
    n = 0
    with a.out.open("a", encoding="utf-8") as out:
        for row in rows:
            rec = process_clip(row, a.source, a.license, a.features_dir, a.retranscribe)
            if rec:
                out.write(json.dumps(rec) + "\n"); n += 1
                print(f"{n:5d} {rec['split']:5s} {rec['speaker_id']:12s} \"{rec['transcript'][:60]}\"")
            if a.limit and n >= a.limit:
                break
    print(f"wrote {n} records -> {a.out}")


if __name__ == "__main__":
    main()
