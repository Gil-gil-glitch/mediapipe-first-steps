"""
Assignment: Rep counter
=======================

Count bicep curls from the camera, using your elbow angle.

    arm straight (angle large)  ->  curl up (angle small)  ->  back down  =  1 rep

This file already runs: it opens the camera, finds your pose and draws your
arm. Your job is to fill in the four TODOs. Search for "TODO" to find them.
Read ASSIGNMENT.md first.

How to run
----------
    python assignment/assignment_starter.py
    python assignment/assignment_starter.py --arm left
    python assignment/assignment_starter.py --video output/my_curls.mp4 --unmirror

Controls
--------
    r          reset the counter
    q or Esc   quit

Useful lessons: 09 (pose), 10 (states and dead zones), 11 (angles), 12 (CSV).
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Assignment - Rep counter"
REPO_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_DIR / "models" / "pose_landmarker_lite.task"
OUTPUT_DIR = REPO_DIR / "output"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# arm -> (shoulder, elbow, wrist) landmark numbers
ARMS = {"left": (11, 13, 15), "right": (12, 14, 16)}
MIN_VISIBILITY = 0.5

# A rep counts when the angle goes below UP_ANGLE and then back above DOWN_ANGLE.
# You may tune these numbers. Explain your choice in your report.
UP_ANGLE = 50
DOWN_ANGLE = 150


class VideoClock:
    """Always-increasing timestamps in milliseconds for VIDEO mode (see lesson 06)."""

    def __init__(self):
        self.start = time.monotonic()
        self.last = -1

    def now_ms(self):
        self.last = max(int((time.monotonic() - self.start) * 1000), self.last + 1)
        return self.last


# ---------------------------------------------------------------------------
# TODO 1: the elbow angle
# ---------------------------------------------------------------------------
def elbow_angle(result, arm):
    """Return the elbow angle of one arm in degrees, or None.

    Args:
        result: A PoseLandmarker result (check it has a person first!).
        arm:    "left" or "right".

    Returns:
        The angle at the elbow between shoulder and wrist (0-180 degrees),
        or None if there is no person or a point's visibility is below
        MIN_VISIBILITY.

    TODO 1:
        - Get the shoulder, elbow and wrist landmarks with ARMS[arm].
        - Return None if any of them isn't visible enough.
        - Compute the angle. Lesson 11 shows how. Choose 2D or 3D and
          explain why in your report.
    """
    return None   # replace this line


# ---------------------------------------------------------------------------
# TODO 2: counting reps
# ---------------------------------------------------------------------------
class RepCounter:
    """Counts reps from a stream of elbow angles.

    It remembers which "stage" the arm is in:
        "down": arm extended, waiting for a curl
        "up":   arm curled, waiting to go back down
    """

    def __init__(self, up_angle=UP_ANGLE, down_angle=DOWN_ANGLE):
        self.up_angle = up_angle
        self.down_angle = down_angle
        self.stage = "down"
        self.reps = 0

    def update(self, angle):
        """Feed in this frame's angle (may be None).

        Returns:
            True if a rep was completed in THIS frame, otherwise False.

        TODO 2:
            - If angle is None, do nothing and return False.
            - "down" -> "up" when the angle drops below self.up_angle.
            - "up" -> "down" when the angle rises above self.down_angle.
              Going back down completes a rep: add 1 to self.reps and return True.
            Why two different angles instead of one? Lesson 10's dead zone is a hint.
        """
        return False   # replace this line

    def reset(self):
        self.stage = "down"
        self.reps = 0


# ---------------------------------------------------------------------------
# TODO 3: show it on screen
# ---------------------------------------------------------------------------
def draw_info(display, angle, counter):
    """Draw the rep count, the stage and the current angle on the picture.

    Args:
        display: The (mirrored) picture to draw on.
        angle:   This frame's elbow angle, or None.
        counter: The RepCounter.

    TODO 3:
        - Show the rep count BIG enough to read from 2 meters away.
        - Show the stage ("up" / "down") and the angle (or "--" if None).
        - Make it look good! A progress bar that fills as the arm curls is a
          nice touch: map angle DOWN_ANGLE -> 0% and UP_ANGLE -> 100%.
    """
    cv2.putText(display, "TODO 3: draw the count here", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


# ---------------------------------------------------------------------------
# TODO 4: save the data
# ---------------------------------------------------------------------------
class RepLogger:
    """Saves time, angle, stage and rep count to a CSV file, one row per frame."""

    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.started_at = time.monotonic()
        self.writer.writerow(["time_s", "angle_deg", "stage", "reps"])

    def write(self, angle, counter):
        """Add one row.

        TODO 4:
            - Write: seconds since started_at, the angle (leave it empty if None),
              counter.stage and counter.reps.
        """
        pass   # replace this line

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# You don't need to change anything below this line (but you may).
# ---------------------------------------------------------------------------
def draw_arm(display, result, arm):
    """Draw the tracked arm on the mirrored display picture."""
    if not result.pose_landmarks:
        return
    landmarks = result.pose_landmarks[0]
    h, w = display.shape[:2]
    points = [(int((1 - landmarks[i].x) * w), int(landmarks[i].y * h)) for i in ARMS[arm]]
    cv2.line(display, points[0], points[1], (255, 255, 255), 4)
    cv2.line(display, points[1], points[2], (255, 255, 255), 4)
    for point in points:
        cv2.circle(display, point, 8, (0, 200, 255), -1)


def parse_args():
    parser = argparse.ArgumentParser(description="Count bicep curls with MediaPipe Pose.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--video", type=Path,
                        help="Use a video file instead of the camera (e.g. one made with lesson 04).")
    parser.add_argument("--unmirror", action="store_true",
                        help="Flip a mirrored video back before detection. Use this with videos "
                             "recorded by lesson 04, which saves the mirrored picture; otherwise "
                             "your left and right arms swap (see lesson 10).")
    parser.add_argument("--arm", choices=["left", "right"], default="right",
                        help="Which of YOUR arms to track. (default: right)")
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
    source = str(args.video) if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source}.")

    clock = VideoClock()
    counter = RepCounter()
    log_path = OUTPUT_DIR / f"reps_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    logger = RepLogger(log_path)
    print(f"Logging to {log_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.unmirror:
                frame = cv2.flip(frame, 1)
            # Detect on the raw picture, show it mirrored (lesson 09).
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), clock.now_ms())
            display = cv2.flip(frame, 1)

            angle = elbow_angle(result, args.arm)
            if counter.update(angle):
                print(f"Rep {counter.reps}!")
            logger.write(angle, counter)

            draw_arm(display, result, args.arm)
            draw_info(display, angle, counter)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                counter.reset()
            elif key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        logger.close()
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print(f"Total reps: {counter.reps}. Data saved to {log_path}")


if __name__ == "__main__":
    main()