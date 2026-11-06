"""Synthetic recordings. Truth stays here and in the UI/evaluation harness."""

from dataclasses import dataclass
import numpy as np

from .estimator import validate_geometry
from .settings import SPEED_OF_SOUND, SAMPLE_RATE, ROOM, MICROPHONES


@dataclass
class Simulation:
    channels: np.ndarray
    sample_rate: int
    true_position: np.ndarray
    emission_time: float
    propagation_samples: np.ndarray
    clean_source: np.ndarray


def simulate(position, microphones=MICROPHONES, room=ROOM, *, seed=7,
             noise=0.0, echo=0.0, sample_rate=SAMPLE_RATE):
    """Render a 12 ms broadband burst in a 120 ms shared recording.

    The burst is a Hann-windowed sum of 80 random-frequency sinusoids.
    Evaluating the continuous formula at t-distance/c gives fractional delays
    without borrowing the estimator's interpolation. Echo: image source in x=0.
    Noise is per-channel white Gaussian standard deviation in waveform units.
    """
    mics, bounds = validate_geometry(microphones, room)
    source = np.asarray(position, dtype=float)
    if source.shape != (2,) or not np.isfinite(source).all() or np.any(source < 0) or np.any(source > bounds):
        raise ValueError("Source must lie inside the room")
    if not np.isfinite([noise, echo]).all() or not 0 <= noise <= 1 or not 0 <= echo <= 1:
        raise ValueError("Noise and echo must be between 0 and 1")
    if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
        raise ValueError("Sample rate must be between 8000 and 96000 Hz")
    rng = np.random.default_rng(seed)
    frequencies = rng.uniform(250, min(6500, sample_rate * 0.30), 80)
    phases = rng.uniform(0, 2 * np.pi, 80)
    emission_time, burst_length = 0.025, 0.012
    # Long enough even for the farthest reflected path in supported rooms.
    duration = max(0.12, emission_time + burst_length + np.linalg.norm(2 * bounds) / SPEED_OF_SOUND + 0.01)
    times = np.arange(int(duration * sample_rate)) / sample_rate

    def waveform(t):
        active = (t >= 0) & (t <= burst_length)
        output = np.zeros_like(t)
        u = t[active]
        envelope = np.sin(np.pi * u / burst_length) ** 2
        output[active] = envelope * np.sin(2 * np.pi * u[:, None] * frequencies + phases).sum(axis=1) / np.sqrt(40)
        return output

    clean = waveform(times - emission_time)
    normalization = max(float(np.max(np.abs(clean))), 1e-12)
    distance = np.linalg.norm(mics - source, axis=1)
    image_source = source * np.array([-1, 1])
    reflected_distance = np.linalg.norm(mics - image_source, axis=1)
    channels = np.empty((len(times), 4))
    for i in range(4):
        channels[:, i] = waveform(times - emission_time - distance[i] / SPEED_OF_SOUND) / max(distance[i], 1.0)
        if echo:
            channels[:, i] += echo * waveform(times - emission_time - reflected_distance[i] / SPEED_OF_SOUND) / max(reflected_distance[i], 1.0)
    channels /= normalization
    channels += rng.normal(0, noise, channels.shape)
    return Simulation(channels, sample_rate, source.copy(), emission_time,
                      distance * sample_rate / SPEED_OF_SOUND, clean / normalization)
