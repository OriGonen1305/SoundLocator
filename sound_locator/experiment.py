"""A frozen held-out experiment, plus a separate runtime benchmark."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter_ns
import numpy as np

from . import __version__
from .delays import estimate_delay
from .estimator import estimate
from .settings import MICROPHONES, LINE_MICROPHONES, ROOM, PAIRS, SPEED_OF_SOUND
from .simulator import simulate
from .solver import solve_position

METHODS = ("threshold", "direct", "fft")
CONDITIONS = (("clean", 0.0, 0.0, MICROPHONES),
              ("noise", 0.035, 0.0, MICROPHONES),
              ("echo", 0.005, 0.9, MICROPHONES),
              ("line_array", 0.003, 0.0, LINE_MICROPHONES),
              ("severe_noise", 0.12, 0.0, MICROPHONES))


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "project_version": __version__}


def source_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate():
    positions = np.random.default_rng(20261105).uniform([0.25, 0.25], [5.75, 4.75], (32, 2))
    trials, summary = [], []
    for condition, noise, echo, mics in CONDITIONS:
        for index, position in enumerate(positions):
            recording = simulate(position, mics, seed=8000 + index, noise=noise, echo=echo)
            for method in METHODS:
                result = estimate(mics, recording.channels, recording.sample_rate, ROOM, method=method)
                error = float(np.linalg.norm(result.position - position) * 100) if result.position is not None else None
                trials.append({"condition": condition, "index": index, "seed": 8000 + index,
                               "source_m": position.tolist(), "method": method,
                               "status": result.status, "error_cm": error,
                               "position_m": result.position.tolist() if result.position is not None else None,
                               "delays_samples": [d.samples for d in result.delays],
                               "residual_samples": result.fit.residual_samples if result.fit else None})
        for method in METHODS:
            rows = [r for r in trials if r["condition"] == condition and r["method"] == method]
            errors = [r["error_cm"] for r in rows if r["error_cm"] is not None]
            summary.append({"condition": condition, "method": method, "trials": len(rows),
                            "accepted": len(errors), "failed": len(rows) - len(errors),
                            "median_cm": float(np.median(errors)) if errors else None,
                            "p90_cm": float(np.percentile(errors, 90)) if errors else None,
                            "accepted_over_25cm": sum(e > 25 for e in errors),
                            "statuses": dict(Counter(r["status"] for r in rows))})
    return {"protocol": "docs/protocol.md v1", "source_sha256": source_hash(),
            "environment": environment(), "position_seed": 20261105,
            "summary": summary, "trials": trials}


def benchmark(repeats=21):
    recording = simulate((2.0, 1.5), seed=7)
    mics = np.asarray(MICROPHONES)
    bounds = [int(np.ceil(np.linalg.norm(mics[j] - mics[i]) * recording.sample_rate / SPEED_OF_SOUND))
              for i, j in PAIRS]
    measured = [estimate_delay(recording.channels[:, i], recording.channels[:, j], bound, "direct").samples
                for (i, j), bound in zip(PAIRS, bounds)]

    def run_delays(method):
        return [estimate_delay(recording.channels[:, i], recording.channels[:, j], bound, method)
                for (i, j), bound in zip(PAIRS, bounds)]

    def measure(function):
        for _ in range(2):
            function()
        times = []
        for _ in range(repeats):
            start = perf_counter_ns()
            function()
            times.append((perf_counter_ns() - start) / 1e6)
        return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
                "samples_ms": times}

    return {"environment": environment(), "source_sha256": source_hash(),
            "samples_per_channel": len(recording.channels), "sample_rate": recording.sample_rate,
            "pairs": 6, "repeats": repeats, "warmups": 2,
            "delay_only": {method: measure(lambda m=method: run_delays(m)) for method in METHODS},
            "solver_only": measure(lambda: solve_position(mics, measured, recording.sample_rate, ROOM)),
            "note": "Warm local timings; excludes simulation, validation, spectral screen, I/O and plotting."}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def summary_table(data):
    lines = ["| Condition | Method | Median cm | P90 cm | Failed / 32 | Accepted >25 cm |",
             "|---|---|---:|---:|---:|---:|"]
    for row in data["summary"]:
        median = "—" if row["median_cm"] is None else f'{row["median_cm"]:.2f}'
        p90 = "—" if row["p90_cm"] is None else f'{row["p90_cm"]:.2f}'
        lines.append(f'| {row["condition"]} | {row["method"]} | {median} | {p90} | {row["failed"]} | {row["accepted_over_25cm"]} |')
    return "\n".join(lines)
