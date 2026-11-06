# From waveforms to position

## Delay convention, with numbers

Let `a` be M1 and `b` be M2. If a short pattern starts at sample 70 in `a`
and at sample 77 in `b`, M1 heard it **first**. Our answer is **+7 samples**:
compare `a[n]` with `b[n+7]`. Reversing the channels gives **−7**.
At 24 kHz, +7 samples means M2 arrived 7/24000 = 0.292 ms later.

For a physical example, M1=(0.5,0.5), M2=(5.5,0.5), source=(2,1.5) m:

```
d1 = sqrt(1.5² + 1²) = 1.802776 m
d2 = sqrt(3.5² + 1²) = 3.640055 m
Δt12 = (d2 - d1) / 343 = +0.00535650 s
Δsamples12 = 24000 × Δt12 = +128.556 samples
```

M1 is closer, so it arrives first. An unknown emission time adds the same
constant to both arrival times and disappears in the subtraction. These
distances illustrate the expected answer; the estimator never receives them.

## Correlation

Subtract each full channel's mean, then for each integer lag k compute

`R[k] = Σ a[n]b[n+k] / sqrt(Σ a[n]² × Σ b[n+k]²)`.

Only overlapping indices enter the sums. Each overlap must retain at least
1% of each channel's centered energy, avoiding spuriously perfect matches of
almost-empty tails. The lag search is bounded by the triangle inequality:
`|Δsamples| ≤ fs × ||mic2 − mic1|| / c`. Use the ceiling for integer search
endpoints, then reject results more than half a sample outside the physical
bound (a quantization allowance). This is a bounded finite-window NCC, with
global DC removal, not a full lag-by-lag Pearson mean calculation.

Pick a positive correlation peak. Reject weak or competing peaks. With adjacent
scores L,C,R, interpolate by `offset = 0.5(L−R)/(L−2C+R)`, clipped to ±0.5 sample.
This is an approximation to a local peak, not exact recovery at every bandwidth.
The amplitude baseline uses the first crossing of 30% of each channel's peak;
noise, reflections and sample phase can move those crossings.

The FFT implementation pads to a power of two ≥2N−1, multiplies
`conj(rfft(a)) * rfft(b)`, takes `irfft`, and selects indices `k mod fft_size`.
Zero-padding prevents circular wraparound. Prefix sums reproduce the overlap
energies of the direct method. No call to `correlate`, convolution, or a
localization library is used. This is ordinary NCC, **not GCC-PHAT**. NumPy
supplies only the [FFT primitives](https://numpy.org/doc/stable/reference/routines.fft.html).

## Position and ambiguity

For candidate p, predict each pair's sample delay:
`fs × (||p−mj|| − ||p−mi||) / 343`. The score is RMS disagreement with six
measured delays. The search starts at 5 cm across the whole room, then refines
locally at 5 mm and 0.5 mm. Bounds include the walls. The displayed heatmap is
the full-room grid on a fixed logarithmic color scale; dark means a small
residual. It is a score, not a likelihood or a confidence interval.

Check for distant near-best candidates across the full grid, and conservatively
reject nearly collinear layouts. With all mics on y=2.5, (x,y) and (x,5−y)
have identical distances. More data from that same array cannot distinguish
those two locations under this model. A heatmap is retained, but no successful
position is displayed. The near-best tolerance and collinearity cutoff are
heuristics, and the grid can miss fine structure. Six pairs are not six
independent measurements, since they share microphones.

## Simulator and its boundary

The simulator evaluates a continuous, Hann-windowed sum of 80 sinusoids at
`t − emission_time − distance/c`, so arrivals can land between samples. It
uses inverse-distance amplitude with a 1 m near-field cap, noise added to each
channel, and optionally an image source at `(-source_x, source_y)` to represent
one specular reflection from x=0. Reflection gain is 0–1 before path attenuation.
This is a 2D horizontal cross-section with all sources and mics at one height;
inverse-distance amplitude is an approximation to 3D spherical spreading in
that plane, not a claim of cylindrical 2D wave propagation.

Seeded synthesis makes runs repeatable. No microphone filtering, quantization
(until WAV export), multiple sources, floor/ceiling paths, clock errors, or
temperature changes are modeled. Noise SD is in waveform units, so distant
microphones have worse SNR. Truth lives in `Simulation`; the UI and experiment
pass only `channels`, microphone positions, rate and bounds to `estimate`.

Independent checks include hand-placed patterns with reversed order, scalar
Python correlation sums, an analytic Gaussian-modulated fractional pulse, and
a 3×4 m rectangle whose distances to (0,0) are 0,3,5,4 m. The independent WAV
fixture uses those integer arrival offsets, not the simulator or project WAV
writer. This prevents a common sign error in simulator and estimator from
validating itself.
