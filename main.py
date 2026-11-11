import argparse
import threading
import time

import cv2
import tkinter as tk
from tkinter import scrolledtext, ttk

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, WINDOW_TITLE
from fall_detector import FallDetector
from hand_motion import HandMotionTracker
from pose_estimator import PoseEstimator
from utils.alerts import play_alert

FEATURES = [
    "Live webcam pose estimation",
    "Skeleton landmark visualization with person bounding box",
    "Posture state classification (standing, walking, sitting, lying, falling)",
    "Fall confirmation duration tracking and confidence score",
    "Visual alert overlay and audio warning support",
    "FPS and status monitoring in the live feed",
    "Configurable camera index and output resolution",
    "Professional Tkinter dashboard with system controls and live telemetry",
    "Left and right wrist movement tracking with visual indicators",
    "Full visible-pose bounding box and key body-joint coordinates",
]


def print_available_features():
    print("Available features:")
    for feature in FEATURES:
        print(f" - {feature}")


class Dashboard(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=18)
        self.app = app
        self.pack(fill="both", expand=True)

        self.configure_style()

        self.status_var = tk.StringVar(value="Idle")
        self.confidence_var = tk.StringVar(value="0%")
        self.fps_var = tk.StringVar(value="0")
        self.alert_var = tk.StringVar(value="Normal")
        self.hands_var = tk.StringVar(value="L: -- | R: --")
        self.body_position_var = tk.StringVar(value="Waiting for body detection")
        self.joints_var = tk.StringVar(value="Joint coordinates will appear here")
        self.camera_var = tk.StringVar(value=f"Camera {app.args.camera_index}")

        self.header = ttk.Frame(self)
        self.header.pack(fill="x", pady=(0, 18))

        ttk.Label(
            self.header,
            text="AI Fall Detection System",
            font=("Segoe UI", 20, "bold"),
            foreground="#0b1f3a",
        ).pack(anchor="w")
        ttk.Label(
            self.header,
            text="Monitoring and alert dashboard",
            font=("Segoe UI", 10),
            foreground="#4d627a",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body)
        sidebar.pack(side="left", fill="y", padx=(0, 18))

        feature_card = ttk.Frame(sidebar, padding=12)
        feature_card.pack(fill="both", expand=True, pady=(0, 12))
        ttk.Label(feature_card, text="Features", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        for feature in FEATURES:
            row = ttk.Frame(feature_card)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text="•", foreground="#0f6db4", font=("Segoe UI", 10, "bold")).pack(side="left")
            ttk.Label(row, text=feature, wraplength=260, justify="left").pack(side="left", padx=(6, 0), anchor="w")

        controls = ttk.Frame(sidebar, padding=12)
        controls.pack(fill="x")
        ttk.Label(controls, text="Controls", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        button_row = ttk.Frame(controls)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="Start", command=self.start_detection, style="Accent.TButton").pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(button_row, text="Stop", command=self.stop_detection).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(button_row, text="Quit", command=self.shutdown).pack(side="left", fill="x", expand=True)

        main = ttk.Frame(body)
        main.pack(side="left", fill="both", expand=True)

        metric_grid = ttk.Frame(main)
        metric_grid.pack(fill="x", pady=(0, 12))

        self.create_metric(metric_grid, "STATUS", self.status_var, "#0f6db4")
        self.create_metric(metric_grid, "CONFIDENCE", self.confidence_var, "#18a957")
        self.create_metric(metric_grid, "FPS", self.fps_var, "#ff9f1c")
        self.create_metric(metric_grid, "ALERT", self.alert_var, "#d93c3c")
        self.create_metric(metric_grid, "HAND MOVEMENT", self.hands_var, "#7952b3")

        slim = ttk.Frame(main)
        slim.pack(fill="x")
        ttk.Label(slim, text="Camera", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(slim, textvariable=self.camera_var, font=("Segoe UI", 10), foreground="#325a88").pack(anchor="w")

        position_panel = ttk.Frame(main, padding=(12, 10))
        position_panel.pack(fill="x", pady=(12, 0))
        ttk.Label(position_panel, text="Body position (preview pixels)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(position_panel, textvariable=self.body_position_var, font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 2))
        ttk.Label(position_panel, textvariable=self.joints_var, font=("Segoe UI", 9), wraplength=590, justify="left").pack(anchor="w")

        log_container = ttk.Frame(main, padding=(0, 14, 0, 0))
        log_container.pack(fill="both", expand=True)
        ttk.Label(log_container, text="Status Log", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
        self.log_area = scrolledtext.ScrolledText(log_container, height=11, wrap=tk.WORD, state="disabled")
        self.log_area.pack(fill="both", expand=True)

        self.append_log("System ready")

    def configure_style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#edf3f9")
        style.configure("TLabel", background="#edf3f9", foreground="#1c2b38")
        style.configure("TButton", padding=(12, 8), font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background="#0f6db4", foreground="white")
        style.map("Accent.TButton", background=[("active", "#0c5b9d")])

    def create_metric(self, parent, label, value_var, accent):
        card = ttk.Frame(parent, padding=12)
        card.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Label(card, text=label, font=("Segoe UI", 9, "bold"), foreground=accent).pack(anchor="w")
        ttk.Label(card, textvariable=value_var, font=("Segoe UI", 18, "bold"), foreground="#0f172a").pack(anchor="w", pady=(4, 0))

    def append_log(self, message):
        self.log_area.configure(state="normal")
        self.log_area.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state="disabled")

    def update_status(self, state, confidence, fps, hand_motion, body_position):
        self.status_var.set(state)
        self.confidence_var.set(f"{confidence}%")
        self.fps_var.set(str(int(fps)))
        self.alert_var.set("Alert" if state == "FALL DETECTED" else "Normal")
        self.hands_var.set(
            f"L: {hand_motion['left'][0]} | R: {hand_motion['right'][0]}"
        )
        center = body_position["center"]
        if center is None:
            self.body_position_var.set("No reliable full-body pose")
            self.joints_var.set("Move into view with shoulders and hips visible")
        else:
            x1, y1, x2, y2 = body_position["bbox"]
            self.body_position_var.set(
                f"Center: ({center[0]}, {center[1]}) | "
                f"Box: x={x1}..{x2}, y={y1}..{y2} | "
                f"Visible joints: {body_position['visible_joints']}/"
                f"{body_position['total_joints']}"
            )
            labels = (
                ("nose", "Head"),
                ("left_shoulder", "L shoulder"),
                ("right_shoulder", "R shoulder"),
                ("left_elbow", "L elbow"),
                ("right_elbow", "R elbow"),
                ("left_wrist", "L wrist"),
                ("right_wrist", "R wrist"),
                ("left_hip", "L hip"),
                ("right_hip", "R hip"),
                ("left_knee", "L knee"),
                ("right_knee", "R knee"),
                ("left_ankle", "L ankle"),
                ("right_ankle", "R ankle"),
            )
            visible_positions = [
                f"{label} {body_position['landmarks'][name]}"
                for name, label in labels
                if body_position["landmarks"].get(name) is not None
            ]
            self.joints_var.set("  |  ".join(visible_positions))

        snapshot = (
            state,
            hand_motion["left"][0],
            hand_motion["right"][0],
            body_position["center"] is not None,
        )
        if snapshot != getattr(self, "_last_logged_snapshot", None):
            self.append_log(
                f"State: {state} | Hands: L {hand_motion['left'][0]}, "
                f"R {hand_motion['right'][0]} | Confidence: {confidence}% | "
                f"Body center: {center if center is not None else 'not detected'}"
            )
            self._last_logged_snapshot = snapshot

    def set_status(self, text):
        self.status_var.set(text)

    def start_detection(self):
        if self.app.processing:
            self.append_log("Detection is already running.")
            return
        self.app.processing = True
        self.app.stop_event.clear()
        self.app.worker = threading.Thread(target=self.app.run_detection_loop, daemon=True)
        self.app.worker.start()
        self.append_log("Started webcam monitoring")

    def stop_detection(self):
        if not self.app.processing:
            self.set_status("Idle")
            self.append_log("Detection is not running.")
            return
        self.app.stop_event.set()
        self.set_status("Stopping")
        self.append_log("Stopping detection...")

    def shutdown(self):
        self.stop_detection()
        self.app.close_window()


class DetectionApp:
    def __init__(self, args):
        self.args = args
        self.processing = False
        self.stop_event = threading.Event()
        self.worker = None
        self.dashboard = None
        self.root = None

    def close_window(self):
        self.stop_event.set()
        if self.root is not None and self.root.winfo_exists():
            self.root.destroy()

    def run_detection_loop(self):
        camera = cv2.VideoCapture(self.args.camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.frame_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.frame_height)

        if not camera.isOpened():
            print(
                f"ERROR: Webcam not detected on index {self.args.camera_index}. "
                "Check your camera settings or use --camera-index."
            )
            self.processing = False
            if self.dashboard is not None:
                self.dashboard.set_status("Camera error")
                self.dashboard.append_log("Camera is unavailable.")
            return

        pose_estimator = PoseEstimator()
        fall_detector = FallDetector()
        hand_tracker = HandMotionTracker()
        previous_state = "STANDING"
        previous_time = time.time()
        fps = 0.0

        colors = {
            "STANDING": (0, 255, 0),
            "WALKING": (0, 255, 0),
            "SITTING": (0, 255, 0),
            "LYING": (0, 165, 255),
            "FALLING": (0, 165, 255),
            "FALL DETECTED": (0, 0, 255),
            "NO PERSON": (255, 255, 255),
        }

        print("AI FALL DETECTION SYSTEM STARTED")
        print_available_features()
        print("Press Q to quit or use the dashboard to stop detection.")

        try:
            while not self.stop_event.is_set():
                success, frame = camera.read()
                if not success:
                    print("Unable to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                results, bbox = pose_estimator.detect(frame)
                landmarks = pose_estimator.get_landmarks(results, frame)
                body_angle = pose_estimator.calculate_body_angle(landmarks)

                state, confidence = fall_detector.analyze(bbox, landmarks, body_angle)
                body_height = bbox[3] - bbox[1] if bbox is not None else frame.shape[0]
                hand_motion = hand_tracker.update(landmarks, body_height, time.time())
                pose_estimator.draw_pose(frame, results)
                body_position = {
                    "center": None,
                    "bbox": bbox,
                    "landmarks": landmarks or {},
                    "visible_joints": 0,
                    "total_joints": len(PoseEstimator.TRACKED_JOINTS),
                }

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    body_position["center"] = ((x1 + x2) // 2, (y1 + y2) // 2)
                    body_position["visible_joints"] = sum(
                        landmarks.get(name) is not None
                        for name in PoseEstimator.TRACKED_JOINTS
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(state, (255, 255, 255)), 2)
                    cv2.putText(
                        frame,
                        f"BODY {body_position['visible_joints']}/{body_position['total_joints']} "
                        f"CENTER {body_position['center']}",
                        (x1, max(22, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        colors.get(state, (255, 255, 255)),
                        2,
                    )

                    for joint_name in PoseEstimator.TRACKED_JOINTS:
                        point = landmarks.get(joint_name)
                        if point is None:
                            continue
                        cv2.circle(frame, point, 4, (255, 255, 0), -1)
                        cv2.putText(
                            frame,
                            f"{joint_name.replace('_', ' ')} {point}",
                            (point[0] + 6, max(14, point[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 0),
                            1,
                        )
                else:
                    guide = (
                        "BODY PARTIALLY VISIBLE - show shoulders and hips"
                        if results is not None
                        else "NO PERSON DETECTED - improve lighting or step into view"
                    )
                    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 58), (20, 20, 20), -1)
                    cv2.putText(
                        frame,
                        guide,
                        (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 200, 255),
                        2,
                    )

                for side, (motion_state, speed, previous_point, point) in hand_motion.items():
                    if point is None:
                        continue
                    color = (0, 0, 255) if motion_state == "MOVING" else (255, 200, 0)
                    cv2.circle(frame, point, 9, color, -1)
                    if motion_state == "MOVING" and previous_point is not None:
                        cv2.arrowedLine(frame, previous_point, point, color, 3, tipLength=0.4)

                if state == "FALL DETECTED" and previous_state != "FALL DETECTED":
                    if not self.args.disable_audio:
                        play_alert()
                    print("WARNING: FALL DETECTED!")
                    if self.dashboard is not None:
                        self.dashboard.append_log("Fall detected! Alert triggered.")

                previous_state = state

                current_time = time.time()
                elapsed = current_time - previous_time
                if elapsed > 0:
                    fps = 1 / elapsed
                previous_time = current_time

                cv2.rectangle(frame, (10, 10), (450, 175), (20, 20, 20), -1)
                cv2.putText(frame, "AI FALL DETECTION SYSTEM", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "STATUS: " + state, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors.get(state, (255, 255, 255)), 2)
                cv2.putText(frame, f"FALL CONFIRMATION: {confidence}%", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {int(fps)}", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"HANDS: L {hand_motion['left'][0]} | R {hand_motion['right'][0]}",
                    (25, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if state == "FALL DETECTED":
                    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 8)
                    cv2.putText(frame, "!!! FALL DETECTED !!!", (max(10, frame.shape[1] // 2 - 210), 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if self.dashboard is not None and self.root is not None and self.root.winfo_exists():
                    self.root.after(
                        0,
                        self.dashboard.update_status,
                        state,
                        confidence,
                        fps,
                        hand_motion,
                        body_position,
                    )

                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            pose_estimator.close()
            cv2.destroyAllWindows()
            self.processing = False
            if self.dashboard is not None:
                self.dashboard.set_status("Stopped")
                self.dashboard.append_log("Detection stopped.")
            print("SYSTEM STOPPED")


def parse_args():
    parser = argparse.ArgumentParser(description="AI Fall Detection System: monitor posture and detect falls from a webcam stream.")
    parser.add_argument("--list-features", action="store_true", help="Display the features available in the system and exit.")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help=f"Webcam index to use (default: {CAMERA_INDEX}).")
    parser.add_argument("--frame-width", type=int, default=FRAME_WIDTH, help=f"Frame width to request from the camera (default: {FRAME_WIDTH}).")
    parser.add_argument("--frame-height", type=int, default=FRAME_HEIGHT, help=f"Frame height to request from the camera (default: {FRAME_HEIGHT}).")
    parser.add_argument("--disable-audio", action="store_true", help="Disable the sound alert when a fall is detected.")
    parser.add_argument("--no-camera", action="store_true", help="Print startup details and exit without opening the webcam.")
    parser.add_argument("--no-gui", action="store_true", help="Run without the Tkinter dashboard; use the standard OpenCV video window only.")
    return parser.parse_args()


def create_dashboard(app):
    root = tk.Tk()
    root.title("AI Fall Detection Dashboard")
    root.geometry("920x640")
    root.minsize(780, 540)
    root.configure(bg="#edf3f9")
    app.root = root

    dashboard = Dashboard(root, app)
    app.dashboard = dashboard
    root.protocol("WM_DELETE_WINDOW", app.close_window)
    return root


def main():
    args = parse_args()

    if args.list_features:
        print_available_features()
        return

    if args.no_camera:
        print("AI FALL DETECTION SYSTEM STARTED")
        print_available_features()
        print("Camera preview is disabled because --no-camera was selected.")
        return

    app = DetectionApp(args)

    if not args.no_gui:
        try:
            root = create_dashboard(app)
            app.dashboard.start_detection()
            root.mainloop()
            return
        except Exception as exc:
            print(f"Tkinter dashboard unavailable ({exc}); falling back to the standard video window.")

    print("AI FALL DETECTION SYSTEM STARTED")
    print_available_features()
    print("Press Q to quit.")
    app.run_detection_loop()


if __name__ == "__main__":
    main()
