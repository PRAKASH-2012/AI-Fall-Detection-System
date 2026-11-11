import math
from collections import deque

from config import (
    FALL_DROP_THRESHOLD,
    FALL_ANGLE_THRESHOLD,
    LYING_ANGLE_THRESHOLD,
    FALL_CONFIRMATION_FRAMES,
    WALKING_MOVEMENT_THRESHOLD
)


class FallDetector:
    def __init__(self):
        self.previous_hip_y = None
        self.previous_center = None
        self.previous_height = None
        self.fall_frames = 0
        self.history = deque(maxlen=10)
        self.current_state = "STANDING"

    def analyze(self, bbox, landmarks, body_angle):
        if bbox is None or landmarks is None:
            self.previous_hip_y = None
            self.previous_center = None
            self.previous_height = None
            self.fall_frames = 0
            self.history.clear()
            self.current_state = "NO PERSON"
            return self.current_state, 0

        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        aspect_ratio = width / height

        left_hip = landmarks["left_hip"]
        right_hip = landmarks["right_hip"]
        if left_hip is None or right_hip is None:
            return self.current_state, 0

        hip_y = (left_hip[1] + right_hip[1]) / 2
        drop = 0.0
        movement = 0.0
        height_drop = 0.0

        if self.previous_hip_y is not None:
            drop = (hip_y - self.previous_hip_y) / max(1, height)

        if self.previous_center is not None:
            movement = math.dist(center, self.previous_center) / max(1, height)

        if self.previous_height is not None:
            height_drop = (
                self.previous_height - height
            ) / max(1, self.previous_height)

        self.previous_hip_y = hip_y
        self.previous_center = center
        self.previous_height = height

        angle = body_angle if body_angle is not None else 0

        falling_condition = (
            drop > FALL_DROP_THRESHOLD
            and (angle > FALL_ANGLE_THRESHOLD or height_drop > 0.12)
        )
        lying_condition = (
            angle > LYING_ANGLE_THRESHOLD or aspect_ratio > 1.15
        )

        if falling_condition:
            self.fall_frames += 1
            state = "FALLING"
        elif lying_condition:
            self.fall_frames += 1
            state = (
                "FALL DETECTED"
                if self.fall_frames >= FALL_CONFIRMATION_FRAMES
                else "LYING"
            )
        else:
            self.fall_frames = 0
            self.history.append(movement)
            average_movement = (
                sum(self.history) / len(self.history) if self.history else 0
            )

            left_knee = landmarks["left_knee"]
            right_knee = landmarks["right_knee"]
            sitting = False

            if left_knee is not None and right_knee is not None:
                knee_y = (left_knee[1] + right_knee[1]) / 2
                sitting = (
                    knee_y - hip_y > height * 0.08
                    and angle < 55
                    and aspect_ratio < 1.15
                )

            if sitting:
                state = "SITTING"
            elif average_movement > WALKING_MOVEMENT_THRESHOLD:
                state = "WALKING"
            else:
                state = "STANDING"

        self.current_state = state
        confidence = min(
            100,
            int(min(1, self.fall_frames / max(1, FALL_CONFIRMATION_FRAMES)) * 100)
        )
        return state, confidence
