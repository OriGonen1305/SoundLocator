"""A small Matplotlib desktop UI; the estimator never receives self.recording."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from .audio import read_geometry, read_wav
from .estimator import estimate, validate_geometry, validate_clip_length
from .settings import MICROPHONES, LINE_MICROPHONES, ROOM, PAIRS, SAMPLE_RATE
from .simulator import simulate

COLORS = ("#2166ac", "#c46c17", "#238b57", "#8a4ea3")


class Demo:
    def __init__(self, wav_path=None, geometry_path=None):
        self.mics, self.room = (read_geometry(geometry_path) if geometry_path else
                               (np.array(MICROPHONES), np.array(ROOM)))
        self.mics = np.array(self.mics, dtype=float, copy=True)
        self.imported = wav_path is not None
        self.channels = None
        self.rate = SAMPLE_RATE
        self.recording = None
        self.result = None
        self.truth_visible = False
        self.dragging = None
        self.next_layout = "line"
        self.input_error = None
        self.method = "direct"
        self.message = "Click inside the room to record a simulated sound."
        self.figure = plt.figure(figsize=(13, 8.2), facecolor="#f7f7f4")
        self.figure.canvas.manager.set_window_title("SoundLocator — four microphones")
        self.figure.text(.045, .95, "SoundLocator", fontsize=22, weight="bold", color="#18334a")
        self.figure.text(.045, .913, "Four microphones. Six time differences. One position estimate.", fontsize=11)
        self.provenance = self.figure.text(.95, .95, "IMPORTED WAV · origin unverified" if self.imported else "SIMULATED DATA",
                                           ha="right", fontsize=11, color="#77541c")
        self.map_ax = self.figure.add_axes([.06, .40, .40, .46])
        self.scale_ax = self.figure.add_axes([.475, .40, .009, .46])
        self.figure.colorbar(ScalarMappable(norm=LogNorm(vmin=.1, vmax=300), cmap="YlGnBu_r"),
                             cax=self.scale_ax, ticks=[.1, 1, 10, 100, 300])
        self.scale_ax.tick_params(labelsize=7)
        self.wave_axes = [self.figure.add_axes([.56, .765 - i * .125, .39, .095]) for i in range(4)]
        self.correlation_ax = self.figure.add_axes([.06, .205, .40, .105])
        self.status_text = self.figure.text(.54, .322, "", fontsize=11, weight="bold", color="#18334a")
        self.error_text = self.figure.text(.54, .288, "", fontsize=10, va="top")
        self.delays_text = self.figure.text(.54, .181, "", fontsize=10, family="monospace", linespacing=1.6)
        self.footer = self.figure.text(.045, .006, "Positive M1→M2 delay: M2 heard the sound later. Units: meters and samples.",
                                       fontsize=9, color="#555555")
        self.predict_button = Button(self.figure.add_axes([.045, .055, .18, .05]), "1  Reveal prediction")
        self.truth_button = Button(self.figure.add_axes([.24, .055, .15, .05]), "2  Reveal truth")
        self.layout_button = Button(self.figure.add_axes([.405, .055, .13, .05]), "Line array")
        self.method_widget = RadioButtons(self.figure.add_axes([.57, .025, .14, .11]),
                                          ("threshold", "direct", "fft"), active=1)
        self.noise_widget = Slider(self.figure.add_axes([.105, .135, .35, .02]), "Noise SD", 0, .15, valinit=0, valfmt="%.3f")
        self.echo_widget = CheckButtons(self.figure.add_axes([.75, .05, .20, .06]), ["Left-wall echo (0.9)"], [False])
        self.predict_button.on_clicked(self.reveal_prediction)
        self.truth_button.on_clicked(self.reveal_truth)
        self.layout_button.on_clicked(self.change_layout)
        self.method_widget.on_clicked(self.change_method)
        self.noise_widget.on_changed(self.change_sound)
        self.echo_widget.on_clicked(self.change_sound)
        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        if self.imported:
            self.channels, self.rate = read_wav(wav_path)
            self.message = "WAV loaded. Reveal prediction; no ground truth available."
            try:
                validate_clip_length(self.mics, len(self.channels), self.rate)
            except ValueError as error:
                self.input_error = str(error)
                self.message = "Cannot estimate this WAV."
                self.predict_button.set_active(False)
            self.truth_button.set_active(False)
            self.layout_button.set_active(False)
            self.noise_widget.set_active(False)
            self.echo_widget.set_active(False)
        self.draw()

    def reset_result(self):
        self.result = None
        self.truth_visible = False

    def record(self, position):
        self.recording = simulate(position, self.mics, self.room, seed=7,
                                  noise=self.noise_widget.val,
                                  echo=.9 if self.echo_widget.get_status()[0] else 0)
        self.channels, self.rate = self.recording.channels, self.recording.sample_rate
        self.reset_result()
        self.message = "Recording ready. Inspect the waves, then reveal prediction."
        self.draw()

    def on_press(self, event):
        if self.map_ax.get_navigate_mode() is not None:
            self.dragging = None
            return
        if self.imported or event.inaxes != self.map_ax or event.button != 1 or event.xdata is None:
            return
        position = np.array([event.xdata, event.ydata])
        if np.any(position < 0) or np.any(position > self.room):
            return
        nearest = int(np.argmin(np.linalg.norm(self.mics - position, axis=1)))
        if np.linalg.norm(self.mics[nearest] - position) < .20:
            self.dragging = nearest
        else:
            self.record(position)

    def on_release(self, event):
        if self.map_ax.get_navigate_mode() is not None:
            self.dragging = None
            return
        if self.dragging is None:
            return
        index, self.dragging = self.dragging, None
        if event.inaxes != self.map_ax or event.xdata is None:
            return
        updated = self.mics.copy()
        updated[index] = [event.xdata, event.ydata]
        try:
            validate_geometry(updated, self.room)
        except ValueError:
            return
        self.mics = updated
        self.change_sound(None)

    def change_sound(self, _):
        if self.imported:
            return
        if self.recording is not None:
            self.record(self.recording.true_position)
        else:
            self.draw()

    def change_layout(self, _):
        if self.imported:
            return
        # A drag changes coordinates, not the action promised by the button.
        preset = LINE_MICROPHONES if self.next_layout == "line" else MICROPHONES
        self.mics = np.array(preset) * self.room / np.array(ROOM)
        self.next_layout = "corners" if self.next_layout == "line" else "line"
        self.layout_button.label.set_text("Line array" if self.next_layout == "line" else "Corner array")
        self.change_sound(None)

    def change_method(self, method):
        self.method = method
        self.reset_result()
        self.message = ("Cannot estimate this WAV." if self.input_error else
                        "Method changed. Reveal prediction on the same recording.")
        self.draw()

    def reveal_prediction(self, _):
        if self.input_error:
            self.message = "Cannot estimate this WAV."
        elif self.channels is None:
            self.message = "Click the room first."
        else:
            self.result = estimate(self.mics, self.channels, self.rate, self.room, method=self.method)
            self.message = f"Status: {self.result.status}"
        self.draw()

    def reveal_truth(self, _):
        if self.recording is not None and self.result is not None:
            self.truth_visible = True
        elif not self.imported:
            self.message = "Reveal prediction before revealing truth."
        self.draw()

    def draw(self):
        ax = self.map_ax
        ax.clear()
        self.scale_ax.set_visible(bool(self.result and self.result.fit))
        if self.result and self.result.fit:
            fit = self.result.fit
            # A fixed logarithmic scale preserves competing minima visually.
            ax.pcolormesh(fit.x, fit.y, np.maximum(fit.scores, .1), shading="auto",
                          norm=LogNorm(vmin=.1, vmax=300), cmap="YlGnBu_r", alpha=.9)
            title = "Heatmap: RMS delay mismatch (samples, log scale)"
        else:
            title = "Click to record · drag a microphone to move it"
            ax.set_facecolor("#edf2f3")
        ax.set_title(title, fontsize=10, pad=10)
        for i, ((x, y), color) in enumerate(zip(self.mics, COLORS)):
            ax.scatter(x, y, c=color, marker="s", s=65, edgecolors="white", zorder=4)
            label_height = -16 if np.ptp(self.mics[:, 1]) < .1 and i % 2 else 8
            ax.annotate(f"M{i+1} ({x:.1f}, {y:.1f})", (x, y), xytext=(5 if x < self.room[0]/2 else -5, label_height),
                        textcoords="offset points", ha="left" if x < self.room[0]/2 else "right", fontsize=8,
                        bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
        if self.result and self.result.position is not None:
            ax.scatter(*self.result.position, marker="x", s=150, linewidths=2.5, c="#d63230", label="Prediction", zorder=6)
        if self.truth_visible:
            ax.scatter(*self.recording.true_position, marker="o", s=190, facecolors="none", edgecolors="#111111",
                       linewidths=1.5, label="Truth", zorder=5)
        if (self.result and self.result.position is not None) or self.truth_visible:
            ax.legend(loc="upper center", ncol=2, fontsize=8)
        ax.set(xlim=(0, self.room[0]), ylim=(0, self.room[1]), xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
        for i, wave_ax in enumerate(self.wave_axes):
            wave_ax.clear()
            wave_ax.set_facecolor("white")
            wave_ax.set_ylabel(f"M{i+1}", color=COLORS[i], rotation=0, labelpad=18)
            wave_ax.tick_params(labelsize=8)
            if self.channels is not None:
                # Shared scale and shared time origin; never align by simulator truth.
                times = np.arange(len(self.channels)) / self.rate * 1000
                stride = max(1, len(times) // 6000)
                wave_ax.plot(times[::stride], self.channels[::stride, i], color=COLORS[i], linewidth=.65)
                limit = max(.05, float(np.max(np.abs(self.channels))) * 1.1)
                wave_ax.set(xlim=(0, times[-1]), ylim=(-limit, limit))
            else:
                wave_ax.set(xlim=(0, 120), ylim=(-1, 1))
            wave_ax.grid(alpha=.15)
            if i < 3:
                wave_ax.tick_params(labelbottom=False)
        self.wave_axes[0].set_title("Synchronized recordings · shared time and amplitude scales", fontsize=10, pad=9)
        self.wave_axes[-1].set_xlabel("Recording time (ms)", fontsize=9)
        self.correlation_ax.clear()
        self.correlation_ax.set_title("M1 → M2 normalized correlation", fontsize=10)
        self.correlation_ax.set_xlabel("Lag (samples) · positive means M2 later", fontsize=9)
        self.correlation_ax.tick_params(labelsize=8)
        self.correlation_ax.set_ylim(-1.05, 1.05)
        if self.result and self.result.delays:
            first = self.result.delays[0]
            if first.lags is not None:
                self.correlation_ax.plot(first.lags, first.scores, color="#2166ac", linewidth=1)
                if first.samples is not None:
                    self.correlation_ax.axvline(first.samples, color="#d63230", linewidth=1)
            else:
                self.correlation_ax.text(.5, .5, "Threshold baseline uses first crossings", ha="center", transform=self.correlation_ax.transAxes, fontsize=9)
            entries = []
            for (i, j), delay in zip(PAIRS, self.result.delays):
                value = f"{delay.samples:+7.2f}" if delay.samples is not None else "    n/a"
                entries.append(f"M{i+1}→M{j+1}: {value}")
            self.delays_text.set_text("\n".join(f"{entries[i]:<19} {entries[i+3]}" for i in range(3)))
        else:
            self.delays_text.set_text("Six pair delays appear after prediction.\nNo truth is passed to the estimator.")
        self.status_text.set_text(self.message)
        detail = self.input_error or ""
        if self.result:
            if self.result.position is not None:
                detail = f"Prediction ({self.result.position[0]:.3f}, {self.result.position[1]:.3f}) m"
            else:
                detail = "No reliable position returned."
            if self.truth_visible:
                truth = self.recording.true_position
                detail += f"   |   Truth ({truth[0]:.2f}, {truth[1]:.2f}) m"
                if self.result.position is not None:
                    error = np.linalg.norm(truth - self.result.position) * 100
                    detail += f"\nError: {error:.2f} cm"
                else:
                    detail += "\nError: unavailable (estimate rejected)"
            elif self.imported:
                detail += "\nTruth and localization error unavailable."
            else:
                detail += "\nTruth remains hidden."
            if self.result.fit:
                detail += f"   |   RMS residual: {self.result.fit.residual_samples:.2f} samples"
        self.error_text.set_text(detail)
        self.figure.canvas.draw_idle()
