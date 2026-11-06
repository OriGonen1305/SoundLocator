"""Re-run the README commands in a source-only copy and a fresh virtualenv.

All generated state stays inside .verification/. The final source ZIP excludes
environments, caches, and scratch data. Run: python3 scripts/verify_clean.py
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".venv", ".cache", ".verification", "scratch", "dist", "build", ".git", "__pycache__"}


def source_files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and
                  not any(part in EXCLUDED or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)
                  and p.suffix not in (".pyc", ".pyo") and p.name != ".DS_Store")


def main():
    work = ROOT / ".verification"
    work.mkdir(exist_ok=True)
    fresh = Path(tempfile.mkdtemp(prefix="clean-", dir=work))
    for path in source_files():
        destination = fresh / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (fresh / ".cache/tmp").mkdir(parents=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(fresh / ".cache/matplotlib"),
               PIP_CACHE_DIR=str(fresh / ".cache/pip"), TMPDIR=str(fresh / ".cache/tmp"))
    records = []
    test_count = None
    commands = [[sys._base_executable, "-m", "venv", ".venv"],
                [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                ["./run.sh", "--test"],
                ["./run.sh", "evaluate", "--output", "scratch/evaluation.json"],
                ["./run.sh", "benchmark", "--output", "scratch/runtime.json"],
                ["./run.sh", "demo", "--snapshot", "scratch/demo.png"],
                ["./run.sh", "simulate", "--source", "2", "1.5", "--seed", "7", "--output", "scratch/simulated.wav"],
                ["./run.sh", "locate", "scratch/simulated.wav", "--geometry", "examples/array.json"],
                ["./run.sh", "demo", "--wav", "scratch/simulated.wav", "--geometry", "examples/array.json",
                 "--snapshot", "scratch/imported.png"]]
    log_path = fresh / "commands.log"
    for command in commands:
        label = " ".join(command if command[0] != sys._base_executable else ["python3", *command[1:]])
        print(f"Running: {label}", flush=True)
        start = time.perf_counter()
        run = subprocess.run(command, cwd=fresh, env=env, text=True, capture_output=True, timeout=240)
        with log_path.open("a") as log:
            log.write(f"$ {label}\n{run.stdout}{run.stderr}\n")
        records.append({"command": label, "exit_code": run.returncode,
                        "elapsed_seconds": round(time.perf_counter() - start, 3)})
        if run.returncode:
            print(run.stdout + run.stderr)
            raise SystemExit(f"Clean check failed; see {log_path}")
        if command == ["./run.sh", "--test"]:
            test_count = int(re.search(r"Ran (\d+) tests", run.stderr).group(1))
    saved = json.loads((ROOT / "results/evaluation.json").read_text())
    replay = json.loads((fresh / "scratch/evaluation.json").read_text())
    if saved["summary"] != replay["summary"] or saved["trials"] != replay["trials"]:
        raise SystemExit("Replayed accuracy results differ from the saved trials")
    # Keep the original evaluation as the baseline when UI/input handling changes.
    # Verify that the replay ran the current package, and record both source hashes.
    digest = hashlib.sha256()
    for path in sorted((ROOT / "sound_locator").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != replay["source_sha256"]:
        raise SystemExit("Clean-copy replay does not describe the current source files")
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_files() if p.relative_to(ROOT) != Path("results/verification.json")}
    report = {"commands": records, "accuracy_replay": "All 480 trial rows and summary values match exactly",
              "environment": replay["environment"], "source_sha256": replay["source_sha256"],
              "baseline_source_sha256": saved["source_sha256"], "test_count": test_count,
              "native_ui": "See docs/verification.md for separately observed desktop checks",
              "hardware_recordings_tested": False, "files_sha256": manifest}
    (ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    archive = ROOT / "dist/soundLocator-source.zip"
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for path in source_files():
            package.write(path, Path("soundLocator") / path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise SystemExit("Source archive failed its integrity check")
        for path in source_files():
            if package.read(str(Path("soundLocator") / path.relative_to(ROOT))) != path.read_bytes():
                raise SystemExit(f"Source archive differs from {path.relative_to(ROOT)}")
    print(f"PASS: {test_count} tests; 480 trials reproduced exactly. Source archive: {archive}")
    print(f"Clean copy and command log: {fresh}")


if __name__ == "__main__":
    main()
