from collections import deque
import math


class HandMotionTracker:
    def __init__(self, threshold=0.45, release_threshold=0.25, window_size=6):
        self.threshold = threshold
        self.release_threshold = release_threshold
        self.tracks = {
            "left": {"previous": None, "samples": deque(maxlen=window_size), "moving": False},
            "right": {"previous": None, "samples": deque(maxlen=window_size), "moving": False},
        }

    def update(self, landmarks, reference_height, timestamp):
        reference_height = max(1, reference_height)
        measurements = {}

        for side in ("left", "right"):
            track = self.tracks[side]
            point = landmarks.get(f"{side}_wrist") if landmarks else None
            previous = track["previous"]

            if point is None:
                track["previous"] = None
                track["samples"].clear()
                track["moving"] = False
                measurements[side] = ("UNAVAILABLE", 0.0, None, None)
                continue

            speed = 0.0
            previous_point = previous[0] if previous else None
            if previous is not None:
                elapsed = timestamp - previous[1]
                if 0 < elapsed <= 0.5:
                    distance = math.dist(point, previous_point) / reference_height
                    track["samples"].append((distance, elapsed))
                    total_time = sum(sample[1] for sample in track["samples"])
                    if total_time > 0:
                        speed = sum(sample[0] for sample in track["samples"]) / total_time
                else:
                    track["samples"].clear()

            track["previous"] = (point, timestamp)

            if speed >= self.threshold:
                track["moving"] = True
            elif speed < self.release_threshold:
                track["moving"] = False

            state = "MOVING" if track["moving"] else "STILL"
            measurements[side] = (state, speed, previous_point, point)

        return measurements
