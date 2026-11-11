import os
import tempfile
from collections import Counter

import cv2
import gradio as gr

from fall_detector import FallDetector
from hand_motion import HandMotionTracker
from pose_estimator import PoseEstimator


MAX_VIDEO_FRAMES = 1800
MAX_FRAME_WIDTH = 960
OUTPUT_DIRECTORY = os.path.join(tempfile.gettempdir(), "fall_detection_outputs")


def _draw_annotations(frame, estimator, results, bbox, landmarks, state, confidence, hands):
    estimator.draw_pose(frame, results)
    colors = {
        "STANDING": (0, 180, 0),
        "WALKING": (0, 180, 0),
        "SITTING": (0, 180, 0),
        "LYING": (0, 165, 255),
        "FALLING": (0, 165, 255),
        "FALL DETECTED": (0, 0, 255),
        "NO PERSON": (255, 255, 255),
    }

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        color = colors.get(state, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, center, 5, (255, 255, 0), -1)
        cv2.putText(
            frame,
            f"BODY CENTER {center}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        for name in estimator.TRACKED_JOINTS:
            point = landmarks.get(name) if landmarks else None
            if point is None:
                continue
            cv2.circle(frame, point, 4, (255, 255, 0), -1)
            cv2.putText(
                frame,
                f"{name.replace('_', ' ')} {point}",
                (point[0] + 5, max(14, point[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 0),
                1,
            )

    for side, (motion, _, previous_point, point) in hands.items():
        if point is None:
            continue
        color = (0, 0, 255) if motion == "MOVING" else (255, 200, 0)
        cv2.circle(frame, point, 8, color, -1)
        if motion == "MOVING" and previous_point is not None:
            cv2.arrowedLine(frame, previous_point, point, color, 2, tipLength=0.4)

    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 520), 76), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"STATUS: {state}  |  FALL CONFIRMATION: {confidence}%",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colors.get(state, (255, 255, 255)),
        2,
    )
    cv2.putText(
        frame,
        f"HANDS: L {hands['left'][0]}  R {hands['right'][0]}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )
    if state == "FALL DETECTED":
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1] - 1, frame.shape[0] - 1),
            (0, 0, 255),
            6,
        )
        cv2.putText(
            frame,
            "FALL DETECTED",
            (max(12, frame.shape[1] // 2 - 120), frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
    return frame


def analyze_video(video_path):
    if not video_path:
        raise gr.Error("Record a webcam clip or choose a video file first.")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise gr.Error("Could not open this video. Try recording or uploading an MP4.")

    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_frames > MAX_VIDEO_FRAMES:
        capture.release()
        raise gr.Error(
            f"This demo accepts clips up to {MAX_VIDEO_FRAMES} frames. "
            "Trim the video and try again."
        )

    estimator = PoseEstimator(model_complexity=1)
    detector = FallDetector()
    hand_tracker = HandMotionTracker()
    frame_counts = Counter()
    detected_frames = 0
    fall_alerts = 0
    previous_state = "NO PERSON"
    latest_position = None
    output_path = None
    writer = None
    processed_frames = 0

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 20.0

        while processed_frames < MAX_VIDEO_FRAMES:
            success, frame = capture.read()
            if not success:
                break

            height, width = frame.shape[:2]
            scale = min(1.0, MAX_FRAME_WIDTH / max(height, width))
            if scale < 1:
                frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            results, bbox = estimator.detect(frame)
            landmarks = estimator.get_landmarks(results, frame)
            angle = estimator.calculate_body_angle(landmarks)
            state, confidence = detector.analyze(bbox, landmarks, angle)
            body_height = bbox[3] - bbox[1] if bbox else frame.shape[0]
            hands = hand_tracker.update(landmarks, body_height, processed_frames / fps)

            if bbox is not None:
                detected_frames += 1
                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                latest_position = (
                    f"Center ({center[0]}, {center[1]}) px; "
                    f"box x={x1}..{x2}, y={y1}..{y2} px"
                )

            frame_counts[state] += 1
            if state == "FALL DETECTED" and previous_state != "FALL DETECTED":
                fall_alerts += 1
            previous_state = state

            frame = _draw_annotations(
                frame, estimator, results, bbox, landmarks, state, confidence, hands
            )

            if writer is None:
                os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
                descriptor, output_path = tempfile.mkstemp(
                    prefix="fall-analysis-", suffix=".mp4", dir=OUTPUT_DIRECTORY
                )
                os.close(descriptor)
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise gr.Error(
                        "Could not create the annotated video on this server."
                    )

            writer.write(frame)
            processed_frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        estimator.close()

    if processed_frames == 0 or output_path is None:
        raise gr.Error("No readable frames were found in that video.")

    duration = processed_frames / fps
    states = ", ".join(
        f"{name}: {count}" for name, count in frame_counts.most_common()
    )
    position = latest_position or "No reliable body position detected."
    summary = (
        f"Processed {processed_frames} frames ({duration:.1f} seconds). "
        f"Body detected in {detected_frames} frames. "
        f"Posture frame counts: {states}. "
        f"Fall alerts: {fall_alerts}. Latest body position: {position}."
    )
    if processed_frames == MAX_VIDEO_FRAMES and reported_frames > MAX_VIDEO_FRAMES:
        summary += f" Analysis stopped at the {MAX_VIDEO_FRAMES}-frame limit."
    return output_path, summary


demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(
        sources=["upload", "webcam"],
        label="Record with your webcam or upload a video (up to 1,800 frames)",
        format="mp4",
    ),
    outputs=[
        gr.Video(label="Annotated video"),
        gr.Textbox(label="Detection summary", lines=4),
    ],
    title="AI Fall Detection System",
    description=(
        "Record a short clip with your browser camera or upload a video. "
        "The app annotates visible body landmarks, wrist movement, posture, "
        "fall alerts, and body position. Video is analyzed after recording; "
        "this is not continuous emergency monitoring."
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
