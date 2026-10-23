"""
Lesson 09: Pose landmarks
=========================

What you'll learn
-----------------
- MediaPipe Pose: 33 landmarks covering the face, arms, body and legs
- Visibility: how sure the model is that a point is actually visible
- Detecting on the camera's real picture, and mirroring only for display
- Two kinds of coordinates: normalized (in the picture) and world (in meters)

The landmarks you'll use most
-----------------------------
     0  nose
    11  left shoulder     12  right shoulder
    13  left elbow        14  right elbow
    15  left wrist        16  right wrist
    23  left hip          24  right hip
    25  left knee         26  right knee
    27  left ankle        28  right ankle

"Left" and "right" are YOUR left and right, not the screen's. (The full list
of 33 names is in LANDMARK_NAMES below.)

Why we don't mirror before detecting (unlike lessons 06-08)
-----------------------------------------------------------
The pose model decides which side is "left" by looking at the picture. In a
mirrored picture your right arm looks like a left arm, so the labels swap.
So here we detect on the raw camera picture and flip only what we SHOW.
(Hand detection is the opposite: it expects a mirrored picture. Lesson 10
lets you see the difference for yourself.)

How to run
----------
    python lessons/part2_mediapipe/09_pose_landmarks.py

Step back so the camera can see at least your upper body.

Controls
--------
    n          show / hide landmark numbers
    p          print all 33 landmarks to the terminal
    q or Esc   quit

Try this
--------
1. Press p while hiding one hand behind your back. What happens to its
   visibility? Does MediaPipe still give it a position?
2. Compare x/y with the world values (wx, wy, wz) for your two wrists.
   Which numbers would you use to measure how far apart your hands are in cm?
3. Draw a line from your nose to the middle of your hips (between 23 and 24).
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Lesson 09 - Pose landmarks"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"

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

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),                     # torso
    (11, 13), (13, 15), (12, 14), (14, 16),                     # arms
    (15, 17), (15, 19), (17, 19), (16, 18), (16, 20), (18, 20), # hands
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),           # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),           # right leg
    (0, 2), (2, 7), (0, 5), (5, 8), (9, 10),                    # face (simplified)
]


class VideoClock:
    """Always-increasing timestamps in milliseconds for VIDEO mode (see lesson 06)."""

    def __init__(self):
        self.start = time.monotonic()
        self.last = -1

    def now_ms(self):
        self.last = max(int((time.monotonic() - self.start) * 1000), self.last + 1)
        return self.last


def to_pixel(landmark, width, height, mirror=True):
    """Convert a normalized landmark to pixel coordinates on the DISPLAYED picture.

    Args:
        landmark:      Has .x and .y between 0.0 and 1.0 (raw camera picture).
        width, height: Size of the displayed picture in pixels.
        mirror:        True if the displayed picture is mirrored, so x must be flipped too.

    Returns:
        (x, y) pixel position as whole numbers.
    """
    x = 1.0 - landmark.x if mirror else landmark.x
    return int(x * width), int(landmark.y * height)


def visibility_color(visibility):
    """Green for a clearly visible point, through yellow, to red for a hidden one (BGR)."""
    v = max(0.0, min(1.0, visibility))
    return (0, int(255 * v), int(255 * (1 - v)))


def draw_pose(frame, landmarks, show_numbers):
    """Draw the skeleton. Points are colored by visibility.

    Args:
        frame:        The (mirrored) display image.
        landmarks:    The 33 landmarks of one person.
        show_numbers: Also write each landmark's number.
    """
    h, w = frame.shape[:2]
    points = [to_pixel(lm, w, h) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if min(landmarks[a].visibility, landmarks[b].visibility) > 0.5:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
    for i, (lm, point) in enumerate(zip(landmarks, points)):
        cv2.circle(frame, point, 5, visibility_color(lm.visibility), -1)
        if show_numbers:
            cv2.putText(frame, str(i), (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def print_landmarks(landmarks, world_landmarks):
    """Print every landmark: picture coordinates, visibility, and world coordinates."""
    print(f"\n {'#':>2} {'name':<17} {'x':>6} {'y':>6} {'z':>6} {'vis':>5}"
          f" | {'wx':>6} {'wy':>6} {'wz':>6}  (world, meters)")
    for i, (lm, wl) in enumerate(zip(landmarks, world_landmarks)):
        print(f" {i:2d} {LANDMARK_NAMES[i]:<17} {lm.x:6.2f} {lm.y:6.2f} {lm.z:6.2f} "
              f"{lm.visibility:5.2f} | {wl.x:6.2f} {wl.y:6.2f} {wl.z:6.2f}")
    print("x, y: position in the camera picture, 0-1 (not mirrored).")
    print("z:    depth, roughly on the same scale as x. Smaller = closer to the camera.")
    print("vis:  0-1, how likely the point is visible (not hidden or out of view).")
    print("world: meters, measured from the point midway between your hips.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Find and draw body pose landmarks with MediaPipe.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="0-1. How sure the model must be before it reports a person. (default: 0.5)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun:  python download_models.py")
    detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=args.min_confidence,
    ))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. See lesson 01 for help.")
    clock = VideoClock()
    show_numbers = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detect on the RAW picture...
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), clock.now_ms())
            # ...and mirror only the picture we show.
            display = cv2.flip(frame, 1)

            # result.pose_landmarks:       one list of 33 landmarks per person
            # result.pose_world_landmarks: the same points in meters
            found = bool(result.pose_landmarks)
            if found:
                landmarks = result.pose_landmarks[0]
                draw_pose(display, landmarks, show_numbers)
                visible = sum(lm.visibility > 0.5 for lm in landmarks)
                status = f"Person found: {visible}/33 landmarks visible"
            else:
                status = "No person found. Step back from the camera."

            cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, "n: numbers   p: print   q: quit", (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("n"):
                show_numbers = not show_numbers
            if key == ord("p"):
                if found:
                    print_landmarks(result.pose_landmarks[0], result.pose_world_landmarks[0])
                else:
                    print("No person in view.")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()