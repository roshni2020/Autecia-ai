# Hashlib for sha256 anonymization of speaker IDs
import hashlib

# OS for audio file deletion
import os

# Secrets for generating a cryptographically random salt
import secrets

# Datetime for session timestamp
from datetime import datetime


def generate_speaker_id() -> str:
    """
    Generate an anonymized speaker ID.
    Formula: sha256(session_timestamp + random_salt)[:12]
    This means no two sessions ever share an ID, and the ID
    cannot be reverse-engineered to identify the speaker.
    """
    # Get current timestamp as a string
    session_timestamp = datetime.utcnow().isoformat()

    # Generate a random salt — different every time
    random_salt = secrets.token_hex(16)

    # Combine timestamp and salt, then hash with sha256
    raw = session_timestamp + random_salt
    hashed = hashlib.sha256(raw.encode()).hexdigest()

    # Return only first 12 characters — short but collision-proof enough
    return hashed[:12]


def delete_audio_if_no_consent(audio_path: str, consent: bool) -> None:
    """
    Delete raw audio file unless speaker explicitly consented to storage.
    This enforces the hard rule:
    'Raw audio deleted after feature extraction unless speaker consented.'
    """
    if not consent:
        # No consent — delete immediately
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"[privacy] Audio deleted (no consent): {audio_path}")
    else:
        # Consent given — keep the file for speaker_declared mode
        print(f"[privacy] Audio retained (consent given): {audio_path}")


def build_session_metadata(mode: str, consent: bool) -> dict:
    """
    Build session metadata block attached to every response.
    Contains anonymized speaker ID and consent status.
    """
    return {
        # Anonymized ID — safe to log, cannot identify speaker
        "speaker_id": generate_speaker_id(),

        # Mode this session ran in
        "mode": mode,

        # Whether speaker consented to audio storage
        "audio_storage_consent": consent,

        # Always research pilot — hardcoded per spec
        "system_stage": "research_pilot",

        # Always true — hardcoded per spec
        "candidate_disclosure_required": True
    }
