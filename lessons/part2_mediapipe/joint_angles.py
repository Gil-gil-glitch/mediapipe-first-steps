"""
Lesson 11: Joint angles
=======================

What you'll learn
-----------------
- Measuring the angle at a joint from three landmarks
  (shoulder - ELBOW - wrist gives the elbow angle)
- The math: vectors and the dot product
- 2D (picture) angles vs 3D (world) angles, and when each one fails

Robots are controlled with joint angles, so this is the bridge between
"the camera sees a person" and "a robot arm moves like a person".

The math
--------
To get the angle at point B between points A and C:

    1. Make two arrows (vectors) starting at B:   BA = A - B,   BC = C - B
    2. The dot product says how much they point the same way:
           BA . BC = |BA| * |BC| * cos(angle)
    3. So:  angle = arccos( (BA . BC) / (|BA| * |BC|) )

    straight arm  -> 180 degrees
    right angle   ->  90 degrees
    fully bent    ->  about 30-40 degrees (your forearm hits your upper arm)

2D or 3D?
---------
2D uses x and y in the picture. It's simple, but it fails when your arm
points toward the camera: a bent arm can look straight from the front.
3D uses MediaPipe's world landmarks (meters), which include depth. It
handles that case better, but the depth is an estimate and is noisier.
Press 3 to switch and compare.

How to run
----------
    python lessons/part2_mediapipe/11_joint_angles.py

Controls
--------
    3          switch between 2D and 3D angles
    q or Esc   quit

Try this
--------
1. Hold your arm straight out to the side, then point it straight at the
   camera. Compare the 2D and 3D elbow angles in both positions.
2. Add the knee angle (hip 23/24 - knee 25/26 - ankle 27/28).
3. Turn the elbow text red when the angle is below 60 degrees.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Lesson 11 - Joint angles"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Each joint: (name, point A, joint B, point C). The angle is measured at B.
JOINTS = [
    ("left elbow",     11, 13, 15),
    ("right elbow",    12, 14, 16),
    ("left shoulder",  23, 11, 13),   # hip - shoulder - elbow: how far the arm is lifted
    ("right shoulder", 24, 12, 14),
]
MIN_VISIBILITY = 0.5


class VideoClock:
    """Always-increasing timestamps in milliseconds for VIDEO mode (see lesson 06)."""

    def __init__(self):
        self.start = time.monotonic()
        self.last = -1

    def now_ms(self):
        self.last = max(int((time.monotonic() - self.start) * 1000), self.last + 1)
        return self.last


def angle_at(a, b, c):
    """Angle at point b, in degrees, between the lines b->a and b->c.

    Works for 2D points (x, y) and 3D points (x, y, z): pass tuples of
    the same length.

    Args:
        a, b, c: Points as tuples of numbers.

    Returns:
        Angle in degrees from 0 to 180, or None if two points are on top of each other.
    """
    ba = [p - q for p, q in zip(a, b)]
    bc = [p - q for p, q in zip(c, b)]
    dot = sum(p * q for p, q in zip(ba, bc))
    length = math.dist(a, b) * math.dist(c, b)
    if length == 0:
        return None
    # Rounding errors can push the value a tiny bit past 1 or -1, which acos refuses.
    cosine = max(-1.0, min(1.0, dot / length))
    return math.degrees(math.acos(cosine))


def joint_angle(result, joint, use_3d, width, height):
    """Compute one joint angle from a pose result.

    Args:
        result:        PoseLandmarker result with at least one person.
        joint:         One entry of JOINTS.
        use_3d:        True = world landmarks (meters), False = picture pixels.
        width, height: Picture size, used to turn 2D coordinates into pixels.

    Returns:
        The angle in degrees, or None if a point isn't visible enough.
    """
    _, a, b, c = joint
    landmarks = result.pose_landmarks[0]
    if min(landmarks[i].visibility for i in (a, b, c)) < MIN_VISIBILITY:
        return None
    if use_3d:
        world = result.pose_world_landmarks[0]
        points = [(world[i].x, world[i].y, world[i].z) for i in (a, b, c)]
    else:
        # Use pixels, not 0-1 values: the picture is wider than it is tall,
        # so 0.1 across is more pixels than 0.1 down, and angles would be distorted.
        points = [(landmarks[i].x * width, landmarks[i].y * height) for i in (a, b, c)]
    return angle_at(*points)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure elbow and shoulder angles.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--start-3d", action="store_true", help="Start in 3D mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun:  python download_models.py")
    detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    ))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. See lesson 01 for help.")
    clock = VideoClock()
    use_3d = args.start_3d

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), clock.now_ms())
            display = cv2.flip(frame, 1)       # detect raw, show mirrored (lesson 09)

            lines = []
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                for joint in JOINTS:
                    name, a, b, c = joint
                    angle = joint_angle(result, joint, use_3d, w, h)
                    lines.append(f"{name:>14}: " + (f"{angle:5.0f} deg" if angle is not None else "   --"))
                    if angle is None:
                        continue
                    # Draw the two lines that form the angle, and the number at the joint.
                    pts = [(int((1 - landmarks[i].x) * w), int(landmarks[i].y * h)) for i in (a, b, c)]
                    cv2.line(display, pts[0], pts[1], (255, 255, 255), 2)
                    cv2.line(display, pts[1], pts[2], (255, 255, 255), 2)
                    cv2.circle(display, pts[1], 8, (0, 200, 255), -1)
                    cv2.putText(display, f"{angle:.0f}", (pts[1][0] + 12, pts[1][1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                lines.append("No person found")

            mode = "3D (world, meters)" if use_3d else "2D (picture pixels)"
            cv2.putText(display, f"Mode: {mode}   3: switch   q: quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, text in enumerate(lines):
                cv2.putText(display, text, (20, 80 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("3"):
                use_3d = not use_3d
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()