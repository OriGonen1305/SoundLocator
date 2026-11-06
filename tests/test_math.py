import math
import unittest
import numpy as np

from sound_locator.delays import direct_correlation, fft_correlation, estimate_delay
from sound_locator.estimator import estimate
from sound_locator.settings import MICROPHONES, ROOM
from sound_locator.solver import solve_position


class DelayTests(unittest.TestCase):
    def test_known_delay_and_reverse(self):
        # Construct placements, not with the simulator or any delay helper.
        a, b = np.zeros(200), np.zeros(200)
        pattern = [0.2, -0.6, 1.0, 0.3, -0.2]
        a[70:75], b[77:82] = pattern, pattern
        for method in ("threshold", "direct", "fft"):
            self.assertAlmostEqual(estimate_delay(a, b, 15, method).samples, 7, places=4)
            self.assertAlmostEqual(estimate_delay(b, a, 15, method).samples, -7, places=4)

    def test_scores_match_independent_scalar_sums(self):
        rng = np.random.default_rng(19)
        a, b = rng.normal(size=(2, 71))
        aa = [float(v) - sum(a) / len(a) for v in a]
        bb = [float(v) - sum(b) / len(b) for v in b]
        expected = []
        for lag in range(-13, 14):
            pairs = [(aa[n], bb[n + lag]) for n in range(len(aa)) if 0 <= n + lag < len(bb)]
            expected.append(sum(x * y for x, y in pairs) /
                            math.sqrt(sum(x*x for x, _ in pairs) * sum(y*y for _, y in pairs)))
        for implementation in (direct_correlation, fft_correlation):
            lags, scores = implementation(a, b, 13)
            np.testing.assert_array_equal(lags, np.arange(-13, 14))
            np.testing.assert_allclose(scores, expected, atol=1e-12)

    def test_fft_padding_and_boundaries(self):
        a, b = np.zeros(101), np.zeros(101)
        a[3:7] = [1, -2, 3, -1]
        b[92:96] = [1, -2, 3, -1]
        for maximum in (0, 9, 90):
            np.testing.assert_allclose(direct_correlation(a, b, maximum)[1],
                                       fft_correlation(a, b, maximum)[1], atol=1e-10)
        self.assertAlmostEqual(estimate_delay(a, b, 90).samples, 89, places=2)

    def test_fractional_delay_independent_continuous_pulse(self):
        # Analytic Gaussian-modulated pulse, unrelated to simulator synthesis.
        t = np.arange(600)
        def pulse(center):
            u = t - center
            return np.exp(-(u / 10) ** 2) * np.cos(0.61 * u)
        for lag in (3.25, -8.4, 0.5):
            for method in ("direct", "fft"):
                result = estimate_delay(pulse(230), pulse(230 + lag), 30, method)
                self.assertEqual(result.status, "ok")
                self.assertAlmostEqual(result.samples, lag, delta=0.035)


class PositionTests(unittest.TestCase):
    def test_center_equal_arrivals(self):
        pattern = np.random.default_rng(44).normal(size=200)
        channels = np.zeros((2400, 4))
        channels[600:800] = pattern[:, None]
        for method in ("threshold", "direct", "fft"):
            result = estimate(MICROPHONES, channels, 24000, ROOM, method=method)
            self.assertEqual(result.status, "ok")
            np.testing.assert_allclose(result.position, [3, 2.5], atol=0.001)
            np.testing.assert_allclose([d.samples for d in result.delays], 0, atol=1e-10)

    def test_off_center_independent_geometry_and_audio(self):
        # Source (0,0), rectangle 3x4: distances exactly 0,3,5,4 m.
        # At 34300 Hz and 343 m/s the arrivals differ by 0,300,500,400 samples.
        mics = [(0, 0), (3, 0), (3, 4), (0, 4)]
        waveform = np.random.default_rng(12).normal(size=180)
        audio = np.zeros((3000, 4))
        for channel, offset in enumerate((0, 300, 500, 400)):
            audio[700 + offset:880 + offset, channel] = waveform
        for method in ("direct", "fft"):
            result = estimate(mics, audio, 34300, (3, 4), method=method)
            self.assertEqual(result.status, "ok")
            np.testing.assert_allclose([d.samples for d in result.delays],
                                       [300, 500, 400, 200, 100, -100], atol=0.002)
            np.testing.assert_allclose(result.position, [0, 0], atol=0.002)

    def test_solver_independent_radical_distances(self):
        # Independent hand calculation for source (1,1) and rectangle 3x4.
        d = [math.sqrt(2), math.sqrt(5), math.sqrt(13), math.sqrt(10)]
        measured = np.array([d[1]-d[0], d[2]-d[0], d[3]-d[0],
                             d[2]-d[1], d[3]-d[1], d[3]-d[2]]) * 24000 / 343
        result = solve_position(np.array([(0, 0), (3, 0), (3, 4), (0, 4)]), measured, 24000, (3, 4))
        self.assertEqual(result.status, "ok")
        np.testing.assert_allclose(result.position, [1, 1], atol=0.001)

    def test_silence_and_missing_channel(self):
        for value in (0, 0.3):
            result = estimate(MICROPHONES, np.full((2400, 4), value), 24000, ROOM)
            self.assertEqual(result.status, "silence")
            self.assertIsNone(result.position)
        audio = np.random.default_rng(9).normal(size=(2400, 4))
        audio[:, 2] = 0
        self.assertEqual(estimate(MICROPHONES, audio, 24000, ROOM).status, "silence")

    def test_periodic_ambiguity_all_methods(self):
        t = np.arange(2400) / 24000
        audio = np.tile(np.sin(2 * np.pi * 1000 * t)[:, None], (1, 4))
        for method in ("threshold", "direct", "fft"):
            result = estimate(MICROPHONES, audio, 24000, ROOM, method=method)
            self.assertEqual(result.status, "ambiguous_signal")
            self.assertIsNone(result.position)

    def test_collinear_layout_mirror_candidates(self):
        mics = np.array([(0.5, 2.5), (2, 2.5), (4, 2.5), (5.5, 2.5)])
        distances = [math.hypot(x - 1.5, y - 1) for x, y in mics]
        measured = [(distances[j] - distances[i]) * 24000 / 343
                    for i, j in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]]
        result = solve_position(mics, measured, 24000, ROOM)
        self.assertEqual(result.status, "ambiguous_layout")
        self.assertIsNone(result.position)
        self.assertLess(result.scores[20, 30], 1e-10)  # (1.5,1)
        self.assertLess(result.scores[80, 30], 1e-10)  # (1.5,4), mirror

    def test_invalid_inputs(self):
        audio = np.zeros((2400, 4))
        for bad in (np.zeros((2400, 3)), audio + np.nan, audio[:3]):
            with self.assertRaises(ValueError):
                estimate(MICROPHONES, bad, 24000, ROOM)
        with self.assertRaises(ValueError):
            estimate([(1,1)] * 4, audio, 24000, ROOM)
