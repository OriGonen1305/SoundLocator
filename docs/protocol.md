# Evaluation protocol (v1)

Frozen before the first held-out evaluation. Development uses analytical tests
and the demo source (2.0, 1.5). No search over algorithm settings is performed.

- Room: 6 by 5 m; four mics 0.5 m in from its corners. 24 kHz, 343 m/s.
- Held-out sources: 32 points from NumPy `default_rng(20261105)`, uniform in
  `[0.25, 5.75] × [0.25, 4.75]`. These sources are not used in development tests.
- Each source uses simulator seed `8000 + index`. Methods get identical audio;
  conditions reuse sources and random draws. No rejected trial is rerun.
- Conditions: clean; noise SD 0.035; left-wall echo gain 0.9 with noise SD 0.005;
  collinear layout with noise SD 0.003; severe noise SD 0.12. Noise is absolute
  waveform amplitude, not a constant SNR. Distance attenuation changes SNR.
- Threshold baseline: first absolute sample above 30% of that channel's peak.
- NCC: global DC removal, overlap energy normalization, physical lag bound,
  positive peak, three-point parabolic refinement. FFT computes the same NCC.
  Each overlap must retain at least 1% of each full channel's centered energy.
- Reject a channel RMS below 1e-7; reject a spectral concentration above 55%
  in five bins; reject NCC peaks below 0.18 or a separate peak ≥94% of the best.
- Full-room grid: 5 cm, refined at 5 mm and 0.5 mm. Residual tolerance 2.5 samples.
  Flag multiple candidates within 1.5 samples of the best and >35 cm away.
  Also conservatively reject array singular-value ratios below 0.025.
- Report median and p90 Euclidean error in cm **only among status `ok`**,
  together with accepted count, all rejected counts by status, and accepted
  errors >25 cm. A failed estimate means status != `ok`; a large accepted error
  is counted separately so filtering cannot hide it. Empty error sets are null.
- Runtime is a separate warm benchmark: same clean 120 ms clip, six pairs,
  two warmups, 21 repeats, `perf_counter_ns`; median and p90. Time the grid solver
  separately. Do not interpret FFT and direct as different accuracy algorithms.

All thresholds are heuristics, not calibrated confidence probabilities.
Freeze this protocol and the settings for comparisons; a future tuning change
requires a new held-out seed and must retain these original results.
