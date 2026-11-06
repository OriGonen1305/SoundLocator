"""Delay estimators. Positive lag means channel b arrived AFTER channel a.

R_ab[k] = sum_n a[n] * b[n+k] / sqrt(sum_n a[n]^2 sum_n b[n+k]^2)
The sums include only overlapping samples. No complete correlation library call.
"""

from dataclasses import dataclass
import numpy as np

from .settings import MIN_CORRELATION, PEAK_RATIO, THRESHOLD_FRACTION


@dataclass
class Delay:
    samples: float | None
    status: str
    peak: float = 0.0
    lags: np.ndarray | None = None
    scores: np.ndarray | None = None


def _prepare(a, b, max_lag):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 8:
        raise ValueError("Expected equal-length 1D channels with at least 8 samples")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Audio must be finite")
    if not isinstance(max_lag, (int, np.integer)) or not 0 <= max_lag < len(a) - 4:
        raise ValueError("Delay bound must leave at least 5 overlapping samples")
    return a - a.mean(), b - b.mean(), np.arange(-max_lag, max_lag + 1)


def direct_correlation(a, b, max_lag):
    """Explicit loop over candidate lags, with overlap energy normalization."""
    a, b, lags = _prepare(a, b, max_lag)
    scores = np.zeros(len(lags))
    n = len(a)
    minimum_a, minimum_b = 0.01 * np.dot(a, a), 0.01 * np.dot(b, b)
    for index, lag in enumerate(lags):
        x = a[max(0, -lag):min(n, n - lag)]
        y = b[max(0, lag):min(n, n + lag)]
        ea, eb = np.dot(x, x), np.dot(y, y)
        denominator = np.sqrt(ea * eb)
        if denominator > 1e-24 and ea >= minimum_a and eb >= minimum_b:
            scores[index] = np.dot(x, y) / denominator
    return lags, scores


def fft_correlation(a, b, max_lag):
    """Same linear correlation, implemented using FFT multiplication.

    Padding to >= 2N-1 prevents circular wraparound. conj(A)*B gives our
    positive-lag convention. Prefix sums reproduce the direct normalization.
    This is ordinary NCC, not GCC-PHAT, so comparison isolates computation.
    """
    a, b, lags = _prepare(a, b, max_lag)
    n = len(a)
    fft_size = 1 << (2 * n - 1).bit_length()
    numerator = np.fft.irfft(np.conj(np.fft.rfft(a, fft_size)) *
                             np.fft.rfft(b, fft_size), fft_size)
    a_energy = np.r_[0.0, np.cumsum(a * a)]
    b_energy = np.r_[0.0, np.cumsum(b * b)]
    ea = a_energy[np.minimum(n, n - lags)] - a_energy[np.maximum(0, -lags)]
    eb = b_energy[np.minimum(n, n + lags)] - b_energy[np.maximum(0, lags)]
    denominator = np.sqrt(np.maximum(ea, 0) * np.maximum(eb, 0))
    valid = (denominator > 1e-24) & (ea >= 0.01 * a_energy[-1]) & (eb >= 0.01 * b_energy[-1])
    scores = np.divide(numerator[lags % fft_size], denominator,
                       out=np.zeros(len(lags)), where=valid)
    return lags, scores


def peak_delay(lags, scores):
    """Reject competing peaks, then fit a parabola to three adjacent scores."""
    k = int(np.argmax(scores))
    peak = float(scores[k])
    if peak < MIN_CORRELATION:
        return Delay(None, "low_correlation", peak, lags, scores)
    maxima = np.flatnonzero((scores[1:-1] >= scores[:-2]) &
                           (scores[1:-1] >= scores[2:])) + 1
    maxima = np.r_[maxima, 0, len(scores) - 1]
    competitors = maxima[np.abs(maxima - k) > 3]
    if len(competitors) and np.max(scores[competitors]) >= PEAK_RATIO * peak:
        return Delay(None, "ambiguous_signal", peak, lags, scores)
    offset = 0.0
    if 0 < k < len(scores) - 1:
        left, center, right = scores[k - 1:k + 2]
        curvature = left - 2 * center + right
        if curvature < -1e-12:
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return Delay(float(lags[k] + offset), "ok", peak, lags, scores)


def estimate_delay(a, b, max_lag, method="direct"):
    if method == "threshold":
        a, b, _ = _prepare(a, b, max_lag)
        arrivals = []
        for channel in (a, b):
            peak = np.max(np.abs(channel))
            if peak < 1e-8:
                return Delay(None, "silence")
            arrivals.append(int(np.flatnonzero(np.abs(channel) >=
                                               THRESHOLD_FRACTION * peak)[0]))
        lag = arrivals[1] - arrivals[0]
        return Delay(float(lag), "ok") if abs(lag) <= max_lag else Delay(None, "inconsistent_delays")
    implementations = {"direct": direct_correlation, "fft": fft_correlation}
    if method not in implementations:
        raise ValueError("Method must be threshold, direct, or fft")
    return peak_delay(*implementations[method](a, b, max_lag))
