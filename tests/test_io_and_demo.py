import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent, NavigationToolbar2
from PIL import Image

from sound_locator.audio import read_geometry, read_wav, write_wav
from sound_locator.demo import Demo
from sound_locator.estimator import estimate
from sound_locator.simulator import simulate
from sound_locator.settings import ROOM, MICROPHONES

REVIEW_CASES = ROOT / "tests/fixtures/final-review"


class FileTests(unittest.TestCase):
    def setUp(self):
        location = ROOT / ".cache" / "tests"
        location.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=location)
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_independent_pcm_wav_to_location_cli(self):
        # This fixture uses integer placement + stdlib WAV, not our simulator
        # or WAV writer. Distances in a 3x4 rectangle are independently 0,3,5,4.
        audio = np.zeros((3000, 4), dtype="<i2")
        pattern = np.random.default_rng(88).integers(-12000, 12000, 150, dtype=np.int16)
        for channel, offset in enumerate((0, 300, 500, 400)):
            audio[700+offset:850+offset, channel] = pattern
        wav_path = self.path / "independent.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setparams((4, 2, 34300, 3000, "NONE", "not compressed"))
            wav.writeframes(audio.tobytes())
        geometry = self.path / "geometry.json"
        geometry.write_text(json.dumps({"room_m": [3, 4], "microphones": [
            {"channel": i, "position_m": p} for i, p in reversed(list(enumerate([(0,0),(3,0),(3,4),(0,4)])))]}))
        result = subprocess.run([sys.executable, "-m", "sound_locator", "locate", str(wav_path),
                                 "--geometry", str(geometry)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "ok")
        np.testing.assert_allclose(data["position_m"], [0, 0], atol=.002)
        np.testing.assert_allclose(data["pair_delay_samples"], [300,500,400,200,100,-100], atol=.002)
        # Check the imported UI can estimate but cannot claim a truth/error.
        app = Demo(wav_path, geometry)
        app.reveal_prediction(None)
        app.reveal_truth(None)
        self.assertEqual(app.result.status, "ok")
        self.assertIsNone(app.recording)
        self.assertFalse(app.truth_visible)
        self.assertIn("unavailable", app.error_text.get_text())
        plt.close(app.figure)

    def test_pcm_roundtrip(self):
        audio = np.random.default_rng(1).uniform(-.5, .5, (600, 4))
        path = self.path / "roundtrip.wav"
        write_wav(path, audio, 24000)
        loaded, rate = read_wav(path)
        self.assertEqual(rate, 24000)
        np.testing.assert_allclose(loaded, audio, atol=4e-5)

    def test_pcm_widths_and_channel_order(self):
        for width in (1, 2, 3, 4):
            path = self.path / f"pcm{width}.wav"
            if width == 1:
                raw = bytes([0, 64, 128, 192]) * 10
            else:
                unit = 2 ** (8 * width - 1)
                raw = b"".join(int(x).to_bytes(width, "little", signed=True)
                               for x in (-unit, -unit//2, 0, unit//2)) * 10
            with wave.open(str(path), "wb") as wav:
                wav.setparams((4, width, 24000, 10, "NONE", "not compressed"))
                wav.writeframes(raw)
            audio, _ = read_wav(path)
            np.testing.assert_array_equal(audio[0], [-1, -.5, 0, .5])

    def test_bad_channel_map(self):
        path = self.path / "bad.json"
        path.write_text(json.dumps({"room_m": ROOM, "microphones": [
            {"channel": 0, "position_m": p} for p in MICROPHONES]}))
        with self.assertRaises(ValueError):
            read_geometry(path)

    def test_cli_silence_and_no_input_overwrite(self):
        path = self.path / "silent.wav"
        write_wav(path, np.zeros((2400, 4)), 24000)
        command = [sys.executable, "-m", "sound_locator", "locate", str(path),
                   "--geometry", str(ROOT / "examples/array.json")]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "silence")
        before = path.read_bytes()
        alias = self.path / "alias.wav"
        alias.hardlink_to(path)
        result = subprocess.run(command + ["--output", str(alias)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(path.read_bytes(), before)

    def test_review_short_wav_gui_message_and_cli_error(self):
        wav = REVIEW_CASES / "too-short.wav"
        geometry = ROOT / "examples/array.json"
        app = Demo(wav, geometry)
        self.addCleanup(plt.close, app.figure)
        self.assertIn("too short", app.error_text.get_text())
        self.assertIn("microphone spacing", app.error_text.get_text())
        self.assertIn("454 samples", app.error_text.get_text())
        self.assertIn("got 200", app.error_text.get_text())
        self.assertFalse(app.predict_button.active)
        for method in ("threshold", "direct", "fft"):
            app.change_method(method)
            app.reveal_prediction(None)
            self.assertIsNone(app.result)
            self.assertFalse(app.truth_visible)
            self.assertEqual(app.message, "Cannot estimate this WAV.")
        result = subprocess.run([sys.executable, "-m", "sound_locator", "locate", str(wav),
                                 "--geometry", str(geometry)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Error: Clip too short", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_review_snapshot_in_one_meter_room(self):
        output = self.path / "small-room.png"
        run = subprocess.run([sys.executable, "-m", "sound_locator", "demo", "--geometry",
                              str(REVIEW_CASES / "one-meter-room.json"), "--snapshot", str(output)],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        with Image.open(output) as png:
            self.assertEqual(png.format, "PNG")
            self.assertGreater(png.width, 500)
            self.assertGreater(png.height, 400)
            png.verify()


class SimulatorTests(unittest.TestCase):
    def test_reproducible_and_fractional_arrivals(self):
        a = simulate((2.0, 1.5), seed=7, noise=.02, echo=.9)
        b = simulate((2.0, 1.5), seed=7, noise=.02, echo=.9)
        np.testing.assert_array_equal(a.channels, b.channels)
        self.assertTrue(np.any(np.abs(a.propagation_samples - np.round(a.propagation_samples)) > .1))

    def test_simulator_against_independent_expected_delay(self):
        recording = simulate((2.0, 1.5), seed=7)
        result = estimate(MICROPHONES, recording.channels, recording.sample_rate, ROOM)
        expected = ((3.5**2 + 1**2)**.5 - (1.5**2 + 1**2)**.5) / 343 * 24000
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.delays[0].samples, expected, delta=.08)
        np.testing.assert_allclose(result.position, [2, 1.5], atol=.002)

    def test_estimator_boundary_has_no_simulator_import(self):
        self.assertEqual(list(inspect.signature(estimate).parameters),
                         ["microphones", "channels", "sample_rate", "room", "method"])
        for module in ("estimator", "delays", "solver"):
            tree = ast.parse((ROOT / f"sound_locator/{module}.py").read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("simulator", node.module or "")


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.app = Demo()

    def tearDown(self):
        plt.close(self.app.figure)

    def click(self, axis, x, y, kind="button_press_event"):
        px, py = axis.transData.transform((x, y))
        event = MouseEvent(kind, self.app.figure.canvas, px, py, button=1)
        self.app.figure.canvas.callbacks.process(kind, event)

    def drag(self, start, end):
        # Use fixed screen coordinates, as a real mouse gesture does.
        first, last = self.app.map_ax.transData.transform([start, end])
        for kind, (x, y) in [("button_press_event", first), ("motion_notify_event", last),
                             ("button_release_event", last)]:
            canvas = self.app.figure.canvas
            buttons = {MouseButton.LEFT} if kind == "motion_notify_event" else None
            canvas.callbacks.process(kind, MouseEvent(kind, canvas, x, y, button=1, buttons=buttons))

    def test_review_layout_label_after_microphone_drag(self):
        self.app.record([2, 1.5])
        self.assertEqual(self.app.layout_button.label.get_text(), "Line array")
        self.app.change_layout(None)
        self.drag([.5, 2.5], [1, 3])
        np.testing.assert_allclose(self.app.mics[0], [1, 3])
        self.assertEqual(self.app.layout_button.label.get_text(), "Corner array")
        self.app.change_layout(None)
        np.testing.assert_allclose(self.app.mics, MICROPHONES)
        self.assertEqual(self.app.layout_button.label.get_text(), "Line array")
        # The next action stays correct after another drag in the corner layout.
        self.drag([.5, .5], [1, 1])
        self.app.change_layout(None)
        np.testing.assert_allclose(self.app.mics[:, 1], 2.5)
        self.assertEqual(self.app.layout_button.label.get_text(), "Corner array")

    def check_navigation_preserves_experiment(self, mode):
        toolbar = NavigationToolbar2(self.app.figure.canvas)
        self.app.record([2, 1.5])
        self.app.reveal_prediction(None)
        self.app.reveal_truth(None)
        source = self.app.recording.true_position.copy()
        microphones, audio = self.app.mics.copy(), self.app.channels.copy()
        recording, channels, result = self.app.recording, self.app.channels, self.app.result
        # Test both empty-map and microphone-origin gestures with the real
        # Matplotlib navigation callbacks, rather than mocking a toolbar flag.
        for start, end in [([3, 3], [4, 4]), ([.5, .5], [1.2, 1.1])]:
            with self.subTest(mode=mode, start=start):
                getattr(toolbar, mode)()
                before = self.app.map_ax.get_xlim(), self.app.map_ax.get_ylim()
                self.drag(start, end)
                self.assertNotEqual(before, (self.app.map_ax.get_xlim(), self.app.map_ax.get_ylim()))
                self.assertIs(self.app.recording, recording)
                self.assertIs(self.app.channels, channels)
                self.assertIs(self.app.result, result)
                np.testing.assert_array_equal(self.app.recording.true_position, source)
                np.testing.assert_array_equal(self.app.mics, microphones)
                np.testing.assert_array_equal(self.app.channels, audio)
                self.assertTrue(self.app.truth_visible)
                self.assertIsNone(self.app.dragging)
                getattr(toolbar, mode)()
                toolbar.home()
        # Normal map actions resume once the toolbar tool is switched off.
        self.click(self.app.map_ax, 4, 3)
        np.testing.assert_allclose(self.app.recording.true_position, [4, 3])
        self.assertIsNone(self.app.result)
        self.assertFalse(self.app.truth_visible)
        self.drag([.5, .5], [1, .6])
        np.testing.assert_allclose(self.app.mics[0], [1, .6])
        # Starting navigation during a pending microphone drag cancels it.
        self.click(self.app.map_ax, 1, .6)
        getattr(toolbar, mode)()
        self.click(self.app.map_ax, 1.5, 1, "button_release_event")
        self.assertIsNone(self.app.dragging)
        np.testing.assert_allclose(self.app.mics[0], [1, .6])
        getattr(toolbar, mode)()

    def test_review_pan_preserves_experiment(self):
        self.check_navigation_preserves_experiment("pan")

    def test_review_zoom_preserves_experiment(self):
        self.check_navigation_preserves_experiment("zoom")

    def test_click_predict_reveal_and_reset(self):
        self.click(self.app.map_ax, 2, 1.5)
        self.assertIsNotNone(self.app.channels)
        self.assertIsNone(self.app.result)
        self.assertFalse(self.app.truth_visible)
        self.app.reveal_truth(None)
        self.assertFalse(self.app.truth_visible)
        with patch("sound_locator.demo.estimate", wraps=estimate) as spy:
            self.app.reveal_prediction(None)
            arguments = spy.call_args.args
            self.assertEqual(len(arguments), 4)
            self.assertIs(arguments[1], self.app.channels)
        self.assertEqual(self.app.result.status, "ok")
        self.assertFalse(self.app.truth_visible)
        self.app.reveal_truth(None)
        self.assertTrue(self.app.truth_visible)
        self.assertIn("Error:", self.app.error_text.get_text())
        self.assertTrue(self.app.scale_ax.get_visible())
        self.click(self.app.map_ax, 4, 3)
        self.assertIsNone(self.app.result)
        self.assertFalse(self.app.truth_visible)

    def test_drag_microphone_and_ambiguous_layout(self):
        self.click(self.app.map_ax, .5, .5)
        self.click(self.app.map_ax, 1, .6, "button_release_event")
        np.testing.assert_allclose(self.app.mics[0], [1, .6])
        self.app.record([2, 1.5])
        self.app.change_layout(None)
        self.app.reveal_prediction(None)
        self.assertEqual(self.app.result.status, "ambiguous_layout")
        self.assertIsNone(self.app.result.position)
        self.assertIsNotNone(self.app.result.fit)
        self.app.reveal_truth(None)
        self.assertIn("unavailable", self.app.error_text.get_text())
        self.app.change_layout(None)
        self.app.reveal_prediction(None)
        self.assertEqual(self.app.result.status, "ok")

    def test_noise_echo_and_method_controls(self):
        self.app.record([2, 1.5])
        before = self.app.channels.copy()
        self.app.noise_widget.set_val(.01)
        self.assertFalse(np.array_equal(before, self.app.channels))
        self.app.echo_widget.set_active(0)
        self.app.method_widget.set_active(2)
        self.assertEqual(self.app.method, "fft")
        self.app.reveal_prediction(None)
        self.assertEqual(self.app.result.status, "ok")
        self.assertFalse(self.app.truth_visible)
