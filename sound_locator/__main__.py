import argparse
import json
import os
from pathlib import Path
import sys
import wave

from .audio import read_geometry, read_wav, write_wav
from .estimator import estimate
from .experiment import evaluate, benchmark, save_json, summary_table
from .settings import ROOM, MICROPHONES
from .simulator import simulate


def same_file(a, b):
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and os.path.samefile(a, b))


def main():
    parser = argparse.ArgumentParser(description="SoundLocator: four synchronized microphones in a 2D room")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Open the click / predict / reveal desktop demo")
    demo.add_argument("--wav", type=Path, help="Open a synchronized 4-channel PCM WAV")
    demo.add_argument("--geometry", type=Path, help="JSON mapping zero-based WAV channels to positions")
    demo.add_argument("--snapshot", type=Path, help="Save the actual demo figure headlessly and exit")
    simulation = sub.add_parser("simulate", help="Write a clearly labeled synthetic WAV")
    simulation.add_argument("--source", nargs=2, type=float, default=[2, 1.5], metavar=("X", "Y"))
    simulation.add_argument("--seed", type=int, default=7)
    simulation.add_argument("--noise", type=float, default=0)
    simulation.add_argument("--echo", type=float, default=0)
    simulation.add_argument("--output", type=Path, default=Path("scratch/simulated.wav"))
    simulation.add_argument("--geometry", type=Path)
    locate = sub.add_parser("locate", help="Estimate from WAV + geometry only")
    locate.add_argument("wav", type=Path)
    locate.add_argument("--geometry", type=Path, required=True)
    locate.add_argument("--method", choices=["threshold", "direct", "fft"], default="direct")
    locate.add_argument("--output", type=Path)
    for name in ("evaluate", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, default=Path(f"scratch/{name}.json"))
    args = parser.parse_args()
    try:
        if args.command == "simulate":
            if args.geometry and same_file(args.output, args.geometry):
                raise ValueError("WAV output must differ from geometry input")
            mics, room = read_geometry(args.geometry) if args.geometry else (MICROPHONES, ROOM)
            recording = simulate(args.source, mics, room, seed=args.seed, noise=args.noise, echo=args.echo)
            write_wav(args.output, recording.channels, recording.sample_rate)
            print(f"SIMULATED: {args.output} | channels 0–3 follow geometry order | {recording.sample_rate} Hz")
        elif args.command == "locate":
            if args.output and any(same_file(args.output, p) for p in (args.wav, args.geometry)):
                raise ValueError("Report output must differ from audio and geometry inputs")
            mics, room = read_geometry(args.geometry)
            channels, rate = read_wav(args.wav)
            result = estimate(mics, channels, rate, room, method=args.method)
            data = {"input": str(args.wav), "provenance": "Imported WAV; physical origin not verified",
                    "method": args.method, **result.as_dict()}
            if args.output:
                save_json(args.output, data)
            print(json.dumps(data, indent=2, allow_nan=False))
            return 0 if result.status == "ok" else 2
        elif args.command == "evaluate":
            data = evaluate()
            save_json(args.output, data)
            print(summary_table(data))
            print(f"Saved all {len(data['trials'])} trials to {args.output}")
        elif args.command == "benchmark":
            data = benchmark()
            save_json(args.output, data)
            print(json.dumps({"delay_only": {k: v["median_ms"] for k, v in data["delay_only"].items()},
                              "solver_only_ms": data["solver_only"]["median_ms"]}, indent=2))
        else:
            if args.wav and not args.geometry:
                raise ValueError("--wav requires --geometry")
            if args.snapshot and any(p and same_file(args.snapshot, p) for p in (args.wav, args.geometry)):
                raise ValueError("Snapshot output must differ from audio and geometry inputs")
            # Keep Matplotlib's generated font cache within this project.
            os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
            if args.snapshot:
                import matplotlib
                matplotlib.use("Agg")
            from .demo import Demo
            app = Demo(wav_path=args.wav, geometry_path=args.geometry)
            if args.snapshot:
                if not args.wav:
                    # Preserve the default demo position, scaled to custom rooms.
                    app.record((app.room[0] / 3, app.room[1] * .3))
                app.reveal_prediction(None)
                if not args.wav:
                    app.reveal_truth(None)
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                app.figure.savefig(args.snapshot, dpi=140)
                print(f"Saved actual demo figure: {args.snapshot}")
            else:
                import matplotlib.pyplot as plt
                plt.show()
        return 0
    except (ValueError, KeyError, TypeError, OSError, wave.Error, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
