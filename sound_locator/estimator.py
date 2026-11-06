"""Public estimator boundary: geometry + recordings + sample rate + room only."""

from dataclasses import dataclass, field
from time import perf_counter
import numpy as np

from .delays import Delay, estimate_delay
from .settings import PAIRS, SPEED_OF_SOUND
from .solver import PositionFit, solve_position


@dataclass
class Estimate:
    status: str
    position: np.ndarray | None = None
    delays: list[Delay] = field(default_factory=list)
    fit: PositionFit | None = None
    delay_ms: float = 0.0
    solver_ms: float = 0.0

    def as_dict(self):
        return {"status": self.status,
                "position_m": self.position.tolist() if self.position is not None else None,
                "pair_delay_samples": [d.samples for d in self.delays],
                "pair_status": [d.status for d in self.delays],
                "residual_samples": self.fit.residual_samples if self.fit else None,
                "delay_ms": self.delay_ms, "solver_ms": self.solver_ms}


def validate_geometry(microphones, room):
    mics, bounds = np.asarray(microphones, dtype=float), np.asarray(room, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or np.any(bounds <= 0) or np.any(bounds > 20):
        raise ValueError("Room must be [width, height], each in (0, 20] meters")
    if mics.shape != (4, 2) or not np.isfinite(mics).all():
        raise ValueError("Exactly four finite [x, y] microphone positions required")
    if np.any(mics < 0) or np.any(mics > bounds):
        raise ValueError("Microphones must lie inside the room")
    if len(np.unique(mics, axis=0)) != 4:
        raise ValueError("Microphone positions must be distinct")
    return mics, bounds


def validate_clip_length(microphones, frame_count, sample_rate):
    """Shared GUI/estimator check for the physically required overlap window."""
    max_required = int(np.ceil(np.linalg.norm(np.ptp(microphones, axis=0)) * sample_rate / SPEED_OF_SOUND))
    minimum = max_required + 5
    if frame_count < minimum:
        raise ValueError(
            "Clip too short for the configured microphone spacing.\n"
            f"Need at least {minimum} samples ({1000 * minimum / sample_rate:.1f} ms); got {frame_count}.\n"
            "Import a longer synchronized four-channel WAV.")
    if frame_count > 2 * sample_rate:
        raise ValueError("Use a synchronized clip no longer than 2 seconds")


def estimate(microphones, channels, sample_rate, room, *, method="direct"):
    """No emission time, true position, simulated delays, or source waveform."""
    microphones, room = validate_geometry(microphones, room)
    channels = np.asarray(channels, dtype=float)
    if method not in ("threshold", "direct", "fft"):
        raise ValueError("Method must be threshold, direct, or fft")
    if not isinstance(sample_rate, (int, np.integer)) or not 8000 <= sample_rate <= 96000:
        raise ValueError("Sample rate must be an integer from 8000 to 96000 Hz")
    if channels.ndim != 2 or channels.shape[1] != 4 or not np.isfinite(channels).all():
        raise ValueError("Audio must have shape (samples, 4) and contain finite values")
    validate_clip_length(microphones, len(channels), sample_rate)
    centered = channels - channels.mean(axis=0)
    if np.min(np.sqrt(np.mean(centered ** 2, axis=0))) < 1e-7:
        return Estimate("silence")
    # A nearly single-frequency recording has many equally plausible cycle shifts.
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    for channel_power in power.T:
        k = int(np.argmax(channel_power))
        concentration = np.sum(channel_power[max(0, k - 2):k + 3]) / channel_power.sum()
        if concentration > 0.55:
            return Estimate("ambiguous_signal")
    start = perf_counter()
    delays = []
    for i, j in PAIRS:
        physical_limit = np.linalg.norm(microphones[j] - microphones[i]) * sample_rate / SPEED_OF_SOUND
        delay = estimate_delay(channels[:, i], channels[:, j], int(np.ceil(physical_limit)), method)
        # Integer search endpoints can be less than one sample beyond the bound.
        if delay.samples is not None and abs(delay.samples) > physical_limit + 0.5:
            delay = Delay(None, "inconsistent_delays", delay.peak, delay.lags, delay.scores)
        delays.append(delay)
    delay_ms = 1000 * (perf_counter() - start)
    invalid = next((d.status for d in delays if d.status != "ok"), None)
    if invalid:
        return Estimate(invalid, delays=delays, delay_ms=delay_ms)
    start = perf_counter()
    fit = solve_position(microphones, [d.samples for d in delays], sample_rate, room)
    return Estimate(fit.status, fit.position, delays, fit, delay_ms,
                    1000 * (perf_counter() - start))
