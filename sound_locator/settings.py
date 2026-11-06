"""Fixed before the held-out evaluation; units are meters, seconds, samples."""

SPEED_OF_SOUND = 343.0
SAMPLE_RATE = 24000
ROOM = (6.0, 5.0)
MICROPHONES = ((0.5, 0.5), (5.5, 0.5), (5.5, 4.5), (0.5, 4.5))
LINE_MICROPHONES = ((0.5, 2.5), (2.0, 2.5), (4.0, 2.5), (5.5, 2.5))
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
GRID_STEP = 0.05
MAX_RESIDUAL_SAMPLES = 2.5
NEAR_BEST_SAMPLES = 1.5
AMBIGUITY_DISTANCE_M = 0.35
MIN_CORRELATION = 0.18
PEAK_RATIO = 0.94
THRESHOLD_FRACTION = 0.30
