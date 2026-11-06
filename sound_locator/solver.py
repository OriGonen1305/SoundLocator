"""Position search using measured pairwise differences, without simulator data."""

from dataclasses import dataclass
import numpy as np

from .settings import (PAIRS, SPEED_OF_SOUND, GRID_STEP, MAX_RESIDUAL_SAMPLES,
                       NEAR_BEST_SAMPLES, AMBIGUITY_DISTANCE_M)


@dataclass
class PositionFit:
    position: np.ndarray | None
    status: str
    residual_samples: float
    x: np.ndarray
    y: np.ndarray
    scores: np.ndarray


def score_positions(points, microphones, measured_samples, sample_rate):
    distances = np.linalg.norm(points[..., None, :] - microphones, axis=-1)
    predicted = np.stack([(distances[..., j] - distances[..., i]) *
                          sample_rate / SPEED_OF_SOUND for i, j in PAIRS], axis=-1)
    return np.sqrt(np.mean((predicted - measured_samples) ** 2, axis=-1))


def solve_position(microphones, measured_samples, sample_rate, room):
    """5 cm full-room grid, then 5 mm and 0.5 mm local grids.

    Heatmap values are RMS residual in samples across six pairs, not a
    probability. The six residuals are correlated because pairs share mics.
    """
    microphones = np.asarray(microphones, dtype=float)
    measured_samples = np.asarray(measured_samples, dtype=float)
    x = np.linspace(0, room[0], int(np.ceil(room[0] / GRID_STEP)) + 1)
    y = np.linspace(0, room[1], int(np.ceil(room[1] / GRID_STEP)) + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx, yy), axis=-1)
    scores = score_positions(points, microphones, measured_samples, sample_rate)
    index = np.unravel_index(np.argmin(scores), scores.shape)
    best = points[index]
    residual = float(scores[index])
    for step in (0.005, 0.0005):
        offsets = np.arange(-10, 11) * step
        rx = np.clip(best[0] + offsets, 0, room[0])
        ry = np.clip(best[1] + offsets, 0, room[1])
        xx, yy = np.meshgrid(rx, ry)
        candidates = np.stack((xx, yy), axis=-1)
        costs = score_positions(candidates, microphones, measured_samples, sample_rate)
        index = np.unravel_index(np.argmin(costs), costs.shape)
        best, residual = candidates[index], float(costs[index])
    singular_values = np.linalg.svd(microphones - microphones.mean(axis=0), compute_uv=False)
    weak_layout = singular_values[1] < 0.025 * max(singular_values[0], 1e-12)
    # Look across the entire room, not just the refinement neighborhood.
    near_best = points[scores <= residual + NEAR_BEST_SAMPLES]
    spread = len(near_best) and np.max(np.linalg.norm(near_best - best, axis=-1)) > AMBIGUITY_DISTANCE_M
    status = "ok"
    if weak_layout or spread:
        status = "ambiguous_layout"
    elif residual > MAX_RESIDUAL_SAMPLES:
        status = "inconsistent_delays"
    return PositionFit(best if status == "ok" else None, status, residual, x, y, scores)
