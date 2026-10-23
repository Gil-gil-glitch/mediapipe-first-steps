"""
Lesson 12: Recording pose data
==============================

What you'll learn
-----------------
- Saving landmark data to a CSV file you can open in Excel, Numbers or Python
- What every number MediaPipe gives you actually means
- Why recorded data is more useful than just watching the screen:
  you can measure, compare and plot it afterwards (lesson 13)

The CSV file
------------
One row per camera frame in which a person was found. Columns:

    time_s                seconds since recording started
    <name>_x, <name>_y    position in the camera picture, 0.0-1.0
                          (x: left -> right of the RAW camera picture, not mirrored;
                           y: top -> bottom, so a raised hand has a SMALLER y)
    <name>_z              depth relative to the hips, roughly the same scale as x.
                          Smaller (more negative) = closer to the camera. An estimate!
    <name>_visibility     0-1, how likely the point is actually visible
    <name>_wx, _wy, _wz   "world" coordinates in METERS, measured from the point
                          midway between your hips. Good for real distances and
                          3D angles. (Try this #2 lets you check which way each axis points.)

<name> is a landmark name like right_wrist. By default we record the arms and
hips (8 landmarks); use --all for all 33.

Frames where no person was found are skipped, so look at time_s to spot gaps.

How to run
----------
    python lessons/part2_mediapipe/12_record_pose_data.py
    python lessons/part2_mediapipe/12_record_pose_data.py --all

Files are saved as output/pose_<date>_<time>.csv

Controls
--------
    r          start / stop recording
    q or Esc   quit (stops any recording first)

Try this
--------
1. Record 10 seconds of waving. Open the CSV in Numbers or Excel and
   make a chart of right_wrist_y. Does the shape match your wave?
2. Record yourself slowly raising your right hand. Does right_wrist_wy go up
   or down as your hand rises? What about right_wrist_y?
3. Add a column with the right elbow angle (copy angle_at from lesson 11).
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Lesson 12 - Record pose data"
REPO_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_DIR / "models" / "pose_landmarker_lite.task"
OUTPUT_DIR = REPO_DIR / "output"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
# Shoulders, elbows, wrists and hips: enough for arm angles.
DEFAULT_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]
VALUES = ["x", "y", "z", "visibility", "wx", "wy", "wz"]


class VideoClock:
    """Always-increasing timestamps in milliseconds for VIDEO mode (see lesson 06)."""

    def __init__(self):
        self.start = time.monotonic()
        self.last = -1

    def now_ms(self):
        self.last = max(int((time.monotonic() - self.start) * 1000), self.last + 1)
        return self.last


class PoseRecorder:
    """Writes pose landmarks to a CSV file, one row per frame.

    Usage:
        recorder = PoseRecorder([11, 12, 13])
        recorder.start(Path("output/test.csv"))
        recorder.write(result)      # once per frame
        recorder.stop()
    """

    def __init__(self, landmark_ids):
        """
        Args:
            landmark_ids: Which of the 33 landmarks to save, e.g. [11, 13, 15].
        """
        self.landmark_ids = landmark_ids
        self.file = None
        self.writer = None
        self.started_at = 0.0
        self.rows = 0

    @property
    def recording(self):
        return self.file is not None

    def header(self):
        """Column names: time_s, then e.g. left_shoulder_x, left_shoulder_y, ..."""
        columns = ["time_s"]
        for i in self.landmark_ids:
            columns += [f"{LANDMARK_NAMES[i]}_{value}" for value in VALUES]
        return columns

    def start(self, path):
        """Open a new CSV file and write the header row."""
        # newline="" stops Windows from adding blank lines between rows.
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.header())
        self.started_at = time.monotonic()
        self.rows = 0

    def write(self, result):
        """Add one row for the first person in a PoseLandmarker result (if any)."""
        if not self.recording or not result.pose_landmarks:
            return
        landmarks = result.pose_landmarks[0]
        world = result.pose_world_landmarks[0]
        row = [round(time.monotonic() - self.started_at, 3)]
        for i in self.landmark_ids:
            lm, wl = landmarks[i], world[i]
            row += [lm.x, lm.y, lm.z, lm.visibility, wl.x, wl.y, wl.z]
        # Round to 4 decimal places: smaller files, and more precision would be noise anyway.
        self.writer.writerow([round(v, 4) for v in row])
        self.rows += 1

    def stop(self):
        """Close the file. Always call this, or the last rows may be lost."""
        if self.file is not None:
            self.file.close()
            self.file = None


def parse_args():
    parser = argparse.ArgumentParser(description="Record pose landmarks to a CSV file.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--all", action="store_true",
                        help="Record all 33 landmarks instead of only arms and hips.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun:  python download_models.py")
    OUTPUT_DIR.mkdir(exist_ok=True)

    detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    ))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. See lesson 01 for help.")

    clock = VideoClock()
    ids = list(range(33)) if args.all else DEFAULT_LANDMARKS
    recorder = PoseRecorder(ids)
    print(f"Recording {len(ids)} landmarks x {len(VALUES)} values = {len(ids) * len(VALUES) + 1} columns.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), clock.now_ms())
            recorder.write(result)

            display = cv2.flip(frame, 1)
            h, w = display.shape[:2]
            if result.pose_landmarks:
                for i in ids:
                    lm = result.pose_landmarks[0][i]
                    cv2.circle(display, (int((1 - lm.x) * w), int(lm.y * h)), 6, (0, 255, 0), -1)

            if recorder.recording:
                seconds = time.monotonic() - recorder.started_at
                cv2.circle(display, (35, 40), 14, (0, 0, 255), -1)
                cv2.putText(display, f"REC {seconds:5.1f} s   {recorder.rows} rows", (60, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(display, "r: start recording   q: quit", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            if not result.pose_landmarks:
                cv2.putText(display, "No person found (frames are skipped)", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                if recorder.recording:
                    recorder.stop()
                    print(f"Saved {recorder.rows} rows.")
                else:
                    path = OUTPUT_DIR / f"pose_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
                    recorder.start(path)
                    print(f"Recording to {path}")
            elif key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if recorder.recording:
            recorder.stop()
            print(f"Saved {recorder.rows} rows.")
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()