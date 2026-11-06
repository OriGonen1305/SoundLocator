# Measured results

All data here is **simulated**. See [protocol](protocol.md) for the fixed settings,
seeds, rejection rules and evaluation definition. 32 held-out positions per
condition; the same recordings are used for all three methods. Accuracy was not
used to tune settings. Clean, noise, echo and line-array cases share source seeds.

The median and p90 include only accepted estimates. Failed means `status != ok`.
`Accepted >25 cm` separately counts large errors that the status checks missed.
A dash means no accepted estimates, not zero error.

| Condition | Method | Median cm | P90 cm | Failed / 32 | Accepted >25 cm |
|---|---|---:|---:|---:|---:|
| clean | threshold | 0.40 | 1.58 | 7 | 0 |
| clean | direct | 0.03 | 0.04 | 0 | 0 |
| clean | fft | 0.03 | 0.04 | 0 | 0 |
| noise | threshold | — | — | 32 | 0 |
| noise | direct | 0.05 | 0.11 | 12 | 0 |
| noise | fft | 0.05 | 0.11 | 12 | 0 |
| echo | threshold | 0.84 | 5.62 | 14 | 0 |
| echo | direct | 0.09 | 0.17 | 5 | 0 |
| echo | fft | 0.09 | 0.17 | 5 | 0 |
| line_array | threshold | — | — | 32 | 0 |
| line_array | direct | — | — | 32 | 0 |
| line_array | fft | — | — | 32 | 0 |
| severe_noise | threshold | 139.57 | 249.56 | 18 | 14 |
| severe_noise | direct | — | — | 32 | 0 |
| severe_noise | fft | — | — | 32 | 0 |

Direct and FFT compute the same NCC. They have matching statuses on every trial;
maximum differences in accepted positions and delays are checked below. Their
comparison is chiefly about runtime, not different localization accuracy.

## Observed failures

- **False success from the threshold baseline:** severe noise, index 1, seed
  8001. Truth `(5.420498, 2.687412)` m; accepted prediction `(2.9405, 2.472)` m.
  Error **248.93 cm**, despite an RMS residual of 2.07 samples. Noise caused
  early crossings that fit a position near the center. The model-consistency
  check alone cannot establish that the measured arrivals belong to the sound.
- **Direct/FFT rejection in noise:** index 3, seed 8003, truth
  `(0.541544, 1.285461)` m, noise SD 0.035. Three pairs involving weaker distant
  signals did not meet the correlation threshold: `low_correlation`.
- **Echo rejection:** the same index/seed with reflection gain 0.9 and noise
  SD 0.005 returned `ambiguous_signal`: a competing correlation peak was too
  close to the best. A direct-path-only solver cannot explain every echo.
- **Baseline failure even when clean:** index 7 returned `inconsistent_delays`,
  RMS residual 2.75 samples. Independently chosen peak-relative crossings can
  change with sampling phase even without added noise.
- **Line array:** all trials were rejected; reflection across the microphone
  line produces an equally plausible source. This is deliberate abstention.

The tiny clean errors come from an ideal continuous broadband source and exact
geometry. They are not evidence of sub-mm real-world localization. Noise is
absolute standard deviation, so different source distances imply different SNRs.
No physical audio recordings were used in this evaluation.

## Separate runtime benchmark

One 120 ms clean clip at 24 kHz (2,880 samples per channel), six pairs; two
warmups and 21 measured repetitions. These local timings exclude simulation,
input validation, the spectral ambiguity screen, file I/O and plotting. The
solver is timed separately, with the same fixed measured delays every time.

| Operation | Median ms | P90 ms |
|---|---:|---:|
| threshold | 0.118 | 0.123 |
| direct | 14.245 | 14.558 |
| fft | 0.904 | 0.917 |
| grid solver | 1.302 | 1.332 |

Environment: macOS-27.0.1-arm64-arm-64bit-Mach-O, Python 3.14.6, NumPy 2.5.4. Timings are machine/load dependent.

Across all 160 direct/FFT trial pairs: maximum accepted coordinate difference 0 m; maximum available delay difference 2.84e-14 samples.

Raw evidence: [480 trials](../results/evaluation.json), [all runtime samples](../results/runtime.json). The JSON also records environment and a SHA-256 of the Python package source.
