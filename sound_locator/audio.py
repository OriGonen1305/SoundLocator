"""Small PCM WAV reader and explicit zero-based channel geometry files."""

import json
from pathlib import Path
import wave
import numpy as np

from .estimator import validate_geometry


def read_geometry(path):
    data = json.loads(Path(path).read_text())
    rows = data["microphones"]
    if len(rows) != 4 or sorted(row["channel"] for row in rows) != [0, 1, 2, 3]:
        raise ValueError("Geometry must map WAV channels 0, 1, 2, 3 exactly once")
    rows = sorted(rows, key=lambda row: row["channel"])
    return validate_geometry([row["position_m"] for row in rows], data["room_m"])


def read_wav(path):
    """Read 8/16/24/32-bit integer PCM. No resampling or channel alignment."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 4 or wav.getcomptype() != "NONE":
            raise ValueError("Expected one uncompressed, synchronized four-channel PCM WAV")
        rate, count, width = wav.getframerate(), wav.getnframes(), wav.getsampwidth()
        if not 8000 <= rate <= 96000 or not 0 < count <= 2 * rate:
            raise ValueError("WAV must be a nonempty clip of at most 2 seconds, at 8–96 kHz")
        raw = wav.readframes(count)
    if len(raw) != count * 4 * width:
        raise ValueError("WAV sample data is truncated")
    if width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(float) - 128) / 128
    elif width in (2, 4):
        values = np.frombuffer(raw, dtype=f"<i{width}").astype(float) / (2 ** (width * 8 - 1))
    elif width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = ((values ^ 0x800000) - 0x800000).astype(float) / 8388608
    else:
        raise ValueError("Supported PCM widths: 8, 16, 24, 32 bits")
    return values.reshape(-1, 4), rate


def write_wav(path, channels, sample_rate):
    """Export PCM16 using one shared gain, preserving relative channel levels."""
    channels = np.asarray(channels, dtype=float)
    if channels.ndim != 2 or channels.shape[1] != 4 or not np.isfinite(channels).all():
        raise ValueError("Expected finite (samples, 4) audio")
    gain = min(1.0, 0.98 / max(float(np.max(np.abs(channels))), 1e-12))
    pcm = np.rint(channels * gain * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(4)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
