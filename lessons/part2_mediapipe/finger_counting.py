"""
Lesson 07: Finger counting
==========================

What you'll learn
-----------------
- Turning landmarks into a decision with simple geometry rules
- Why "is the tip above the knuckle?" breaks when you turn your hand,
  and how comparing DISTANCES fixes it
- Counting fingers on two hands (0-10)

MediaPipe only gives you points. Deciding what the points MEAN is up to you.
This is the most important idea in Part 2.

The rule we use
---------------
A finger is extended when its tip is farther from the wrist than its middle
joint is. When you curl a finger, the tip comes back toward the wrist.

    extended:   wrist ............ middle joint ...... tip      (tip is far)
    curled:     wrist ....... tip  middle joint                 (tip is close)

The thumb bends sideways across the palm instead, so for the thumb we measure
distances to the base of the pinky (landmark 17), and compare the tip with
the thumb's knuckle (landmark 2): a folded thumb's tip ends up closer to the
pinky than its own knuckle is.

Distances don't change when you rotate your hand, so this rule works with
your hand upright, sideways or upside down.

How to run
----------
    python lessons/part2_mediapipe/07_finger_counting.py
    python lessons/part2_mediapipe/07_finger_counting.py --ratio 1.2

Controls
--------
    q or Esc   quit

Try this
--------
1. Run with --ratio 1.0 and then --ratio 1.4. Which fingers get miscounted?
2. Replace is_extended() with the simple rule "tip.y < middle_joint.y"
   (tip higher on screen). Count three fingers, then turn your hand sideways.
3. Recognise a "peace sign": only index and middle extended. Show "PEACE!".
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Lesson 07 - Finger counting"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

WRIST = 0
PINKY_BASE = 17
# finger name -> (joint landmark to compare with, tip landmark)
FINGERS = {
    "thumb":  (2, 4),     # the thumb's knuckle (see "The rule" above)
    "index":  (6, 8),
    "middle": (10, 12),
    "ring":   (14, 16),
    "pinky":  (18, 20),
}


class VideoClock:
    """Always-increasing timestamps in milliseconds for VIDEO mode (see lesson 06)."""

    def __init__(self):
        self.start = time.monotonic()
        self.last = -1

    def now_ms(self):
        self.last = max(int((time.monotonic() - self.start) * 1000), self.last + 1)
        return self.last


def distance(a, b):
    """Straight-line distance between two landmarks, in normalized units.

    We only compare distances with each other, so the units don't matter.
    """
    return math.hypot(a.x - b.x, a.y - b.y)


def is_extended(landmarks, finger, ratio):
    """Decide whether one finger is extended.

    Args:
        landmarks: The 21 landmarks of one hand.
        finger:    A key of FINGERS, e.g. "index".
        ratio:     How much farther the tip must be than the joint.
                   1.0 = any amount; bigger = the finger must be straighter.
                   A straight finger's tip is only about 1.3x farther than its
                   middle joint, so keep this between 1.0 and 1.3.

    Returns:
        True if the finger counts as extended.
    """
    joint_id, tip_id = FINGERS[finger]
    anchor = landmarks[PINKY_BASE if finger == "thumb" else WRIST]
    return distance(landmarks[tip_id], anchor) > ratio * distance(landmarks[joint_id], anchor)


def count_fingers(landmarks, ratio):
    """Return a dict {finger name: True/False} for one hand."""
    return {finger: is_extended(landmarks, finger, ratio) for finger in FINGERS}


def draw_hand(frame, landmarks, fingers):
    """Mark each fingertip: green circle if extended, small grey dot if not."""
    h, w = frame.shape[:2]
    for finger, extended in fingers.items():
        tip = landmarks[FINGERS[finger][1]]
        point = (int(tip.x * w), int(tip.y * h))
        if extended:
            cv2.circle(frame, point, 14, (0, 220, 0), 3)
        else:
            cv2.circle(frame, point, 5, (160, 160, 160), -1)


def parse_args():
    parser = argparse.ArgumentParser(description="Count raised fingers on up to two hands.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--ratio", type=float, default=1.1,
                        help="How much farther a tip must be than its joint "
                             "to count as extended. (default: 1.1)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun:  python download_models.py")
    detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    ))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. See lesson 01 for help.")
    clock = VideoClock()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), clock.now_ms())

            total = 0
            lines = []
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                fingers = count_fingers(landmarks, args.ratio)
                draw_hand(frame, landmarks, fingers)
                count = sum(fingers.values())        # True counts as 1, False as 0
                total += count
                up = [name for name, extended in fingers.items() if extended]
                lines.append(f"{handedness[0].category_name}: {count}  {', '.join(up)}")

            # Big number in the corner, details underneath.
            cv2.putText(frame, str(total), (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 220, 0), 8)
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (20, 160 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
            if not lines:
                cv2.putText(frame, "Show me your hands", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()