import math

import cv2
import mediapipe as mp


class PoseEstimator:
    MIN_LANDMARK_VISIBILITY = 0.35
    TRACKED_JOINTS = (
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    def __init__(self, model_complexity=2, static_image_mode=False):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, None

        landmarks = self.get_landmarks(results, frame)
        if landmarks is None:
            return results, None

        torso_visible = any(
            landmarks[name] is not None
            for name in ("left_shoulder", "right_shoulder")
        ) and any(
            landmarks[name] is not None for name in ("left_hip", "right_hip")
        )
        points = [
            point for point in landmarks.values() if point is not None
        ]
        if not torso_visible or len(points) < 2:
            return results, None

        h, w = frame.shape[:2]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        padding = max(4, int((max(ys) - min(ys)) * 0.04))
        bbox = (
            max(0, min(xs) - padding),
            max(0, min(ys) - padding),
            min(w - 1, max(xs) + padding),
            min(h - 1, max(ys) + padding),
        )
        return results, bbox

    def draw_pose(self, frame, results):
        if results and results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

    def get_landmarks(self, results, frame):
        if results is None or results.pose_landmarks is None:
            return None

        h, w = frame.shape[:2]
        pose_landmarks = results.pose_landmarks.landmark
        landmarks = {}

        for landmark_type in self.mp_pose.PoseLandmark:
            name = landmark_type.name.lower()
            landmark = pose_landmarks[landmark_type.value]
            if (
                landmark.visibility < self.MIN_LANDMARK_VISIBILITY
                or not math.isfinite(landmark.x)
                or not math.isfinite(landmark.y)
                or not 0 <= landmark.x <= 1
                or not 0 <= landmark.y <= 1
            ):
                landmarks[name] = None
                continue

            x = min(w - 1, int(landmark.x * w))
            y = min(h - 1, int(landmark.y * h))
            landmarks[name] = (x, y)

        return landmarks

    def calculate_body_angle(self, landmarks):
        if landmarks is None:
            return None

        ls = landmarks["left_shoulder"]
        rs = landmarks["right_shoulder"]
        lh = landmarks["left_hip"]
        rh = landmarks["right_hip"]

        if any(p is None for p in (ls, rs, lh, rh)):
            return None

        shoulder_x = (ls[0] + rs[0]) / 2
        shoulder_y = (ls[1] + rs[1]) / 2
        hip_x = (lh[0] + rh[0]) / 2
        hip_y = (lh[1] + rh[1]) / 2

        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y
        return abs(math.degrees(math.atan2(dx, -dy)))

    def close(self):
        self.pose.close()
