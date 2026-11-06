# Local verification

Tested on Apple Silicon macOS, Python 3.14.6, NumPy 2.5.4, Matplotlib 3.11.2.

`./run.sh --test` runs 27 tests. The important independent references are:

- A pattern manually placed seven samples later, then reversed.
- Correlation scores computed with scalar Python sums, including the sign.
- An analytic Gaussian-modulated pulse shifted by fractional samples.
- A 3×4 m rectangle: known distances 0,3,5,4 m from one corner, with manually
  placed arrivals at 34300 Hz (100 samples per meter).
- A WAV built with the standard-library writer and those same independent
  integer offsets; the actual CLI reads it and estimates the source.
- Equal simultaneous arrivals at a symmetric array's center, and two mirror
  positions around a collinear array. Silence and a periodic tone return no point.

Tests also cover PCM formats and channel mapping, simulator reproducibility,
the estimator's input boundary, GUI mouse events, microphone dragging, reset
behavior, echo/noise controls, and imported audio with no truth available.

## Final-review fixes (2026-11-06)

All four supplied review issues were reproduced before editing. The original
`one-meter-room.json` and `too-short.wav` are retained unchanged under
`tests/fixtures/final-review/`. Five new regression tests cover:

- Line array → drag one mic off the line → Corner array in one click, with
  the label continuing to describe the next action after subsequent drags.
- Pan and Zoom separately, with gestures starting both on empty map space
  and on microphones. The tests use Matplotlib's actual navigation callbacks,
  confirm changed view limits, and assert unchanged source, microphone positions,
  audio samples, recording/result objects, and revealed truth. Normal source
  clicks and microphone drags resume after switching the tool off.
- The 200-frame WAV reports that the configured array needs at least 454 samples
  (18.9 ms at 24 kHz), disables prediction, and stays usable. The `locate` CLI
  still returns exit code 1 with an error message and no traceback.
- The custom-room snapshot command succeeds and produces a decoded, verified
  PNG with a demonstration source inside the supplied 1×1 m room.

For these fixes, the native desktop window was opened and the short-WAV message
was visually checked. The Pan toolbar toggle responded through accessibility.
However, native coordinate clicks/drags repeatedly failed in the computer-use
tool with `noWindowsAvailable`, even after relaunching the app. The corrected
drag/layout/Pan/Zoom sequences were therefore verified **through automated
tests only**, not manually in this review. The earlier native smoke checks
described below predate these fixes. The custom-room PNG was visually inspected.

The native Matplotlib window was opened on macOS and operated with real UI
clicks: source click → four waveforms → prediction with truth hidden → reveal
truth and error. The line-array button produced two dark score minima and
`ambiguous_layout` without an estimated point. Switching back, enabling echo,
adding noise and selecting FFT produced an estimate and a displayed 0.12 cm
error in that one simulated demonstration. This is a smoke check, not a
hardware accuracy measurement. Exported UI images were visually inspected.

## Clean-copy replay

```sh
python3 scripts/verify_clean.py
```

This copies source files without `.venv`, caches, scratch data or verification
directories, creates a new environment, installs the pinned requirements, and
runs every noninteractive README command. It also renders simulator and WAV
views with the actual UI class. All **480 trial rows and summary values** must
match the saved evaluation exactly. The interactive imported-WAV view is
covered with the Agg backend; native interaction limits for this review are
described above. The original evaluation and runtime JSON files are preserved.
The verification report records both the original evaluation source hash and
the current replay source hash, since UI/input validation fixes change the
package hash without changing the 480 numerical trial rows or summaries.

[verification.json](../results/verification.json) records command exit codes,
the environment, and file hashes. Disposable environments/logs stay under the
ignored `.verification/` folder. A source-only archive is written to
`dist/soundLocator-source.zip`; neither local environments nor generated caches
are included. The script does not commit, publish, or push.

No physical microphone recording, Linux/Windows runtime, or hosted CI run was
tested. The statistical evaluation is entirely synthetic. Re-running runtime
benchmarks will change timings; the accuracy replay compares numerical trials,
not wall-clock performance.
