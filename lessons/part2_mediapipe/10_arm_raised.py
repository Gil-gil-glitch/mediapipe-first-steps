"""
Lesson 10: Which arm is raised?
===============================

What you'll learn
-----------------
- Turning pose landmarks into a simple yes/no answer: "is this arm up?"
- In picture coordinates y grows DOWNWARD, so "higher" means a SMALLER y
- A "dead zone" so the answer doesn't flicker when your hand is near the line
- The mirror trap: what happens if you mirror the picture before detecting

The rule
--------
    wrist clearly ABOVE the shoulder  -> UP
    wrist clearly BELOW the shoulder  -> DOWN
    in between (the dead zone)        -> keep the previous answer
    wrist or shoulder not visible     -> ?

Without the dead zone, a hand held right at shoulder height would switch
between UP and DOWN many times per second.

How to run
----------
    python lessons/part2_mediapipe/10_arm_raised.py
    python lessons/part2_mediapipe/10_arm_raised.py --margin 0.1
    python lessons/part2_mediapipe/10_arm_raised.py --mirror-before-detect   # the trap!

Controls
--------
    q or Esc   quit

Try this
--------
1. Run with --mirror-before-detect and raise only your RIGHT arm.
   What does the screen say? Explain why, using lesson 09.
2. Set --margin 0 and hold your wrist at shoulder height. What happens?
3. Count how many times each arm has been raised and show the counts.
4. Add a new state "OUT" for an arm held out to the side at shoulder height.
   (Hint: compare the x distance between wrist and shoulder too.)
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Lesson 10 - Arm raised"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# arm -> (shoulder, elbow, wrist) landmark numbers
ARMS = {
    "left":  (11, 13, 15),
    "right": (12, 14, 16),
}
MIN_VISIBILITY = 0.5

UP_COLOR, DOWN_COLOR, UNKNOWN_COLOR = (0, 220, 0), (0, 140, 255), (150, 150, 150)


class VideoClock:
    """Always-increasing timestamps in milliseconds for VIDEO mode (see lesson 06)."""

    def __init__(self):
        self.start = time.monotonic()
        self.last = -1

    def now_ms(self):
        self.last = max(int((time.monotonic() - self.start) * 1000), self.last + 1)
        return self.last


def arm_state(landmarks, arm, previous, margin):
    """Decide whether one arm is UP, DOWN, or unknown ("?").

    Args:
        landmarks: The 33 pose landmarks.
        arm:       "left" or "right".
        previous:  This arm's state in the last frame (used inside the dead zone).
        margin:    Size of the dead zone, as a fraction of the picture height.

    Returns:
        "UP", "DOWN" or "?".
    """
    shoulder_id, _, wrist_id = ARMS[arm]
    shoulder, wrist = landmarks[shoulder_id], landmarks[wrist_id]
    if min(shoulder.visibility, wrist.visibility) < MIN_VISIBILITY:
        return "?"

    # Remember: y = 0 is the TOP of the picture. Higher on screen = smaller y.
    if wrist.y < shoulder.y - margin:
        return "UP"
    if wrist.y > shoulder.y + margin:
        return "DOWN"
    return previous if previous != "?" else "DOWN"     # inside the dead zone


def draw_arm(display, landmarks, arm, state, mirrored):
    """Draw one arm, colored by its state.

    Args:
        display:  The image being shown (modified in place).
        landmarks: The 33 pose landmarks (from the raw camera picture).
        arm:      "left" or "right".
        state:    "UP", "DOWN" or "?".
        mirrored: Whether `display` is a mirrored copy of the detected picture.
    """
    h, w = display.shape[:2]
    color = {"UP": UP_COLOR, "DOWN": DOWN_COLOR}.get(state, UNKNOWN_COLOR)
    points = []
    for i in ARMS[arm]:
        x = 1.0 - landmarks[i].x if mirrored else landmarks[i].x
        points.append((int(x * w), int(landmarks[i].y * h)))
    cv2.line(display, points[0], points[1], color, 6)
    cv2.line(display, points[1], points[2], color, 6)
    for point in points:
        cv2.circle(display, point, 8, (255, 255, 255), -1)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect which arm is raised.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Dead zone around shoulder height, as a fraction of "
                             "the picture height. (default: 0.05)")
    parser.add_argument("--mirror-before-detect", action="store_true",
                        help="Mirror the picture BEFORE pose detection (the wrong way!) "
                             "to see the left/right labels swap.")
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
    states = {"left": "?", "right": "?"}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror_before_detect:
                frame = cv2.flip(frame, 1)       # the wrong way, for the experiment
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), clock.now_ms())
            # Show a mirrored picture either way, so the screen feels like a mirror.
            mirrored = not args.mirror_before_detect
            display = cv2.flip(frame, 1) if mirrored else frame

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                for arm in ARMS:
                    new_state = arm_state(landmarks, arm, states[arm], args.margin)
                    if new_state != states[arm] and new_state != "?":
                        print(f"{time.strftime('%H:%M:%S')}  {arm} arm {new_state}")
                    states[arm] = new_state
                    draw_arm(display, landmarks, arm, new_state, mirrored)
            else:
                states = {"left": "?", "right": "?"}

            # Your left arm appears on the LEFT of a mirror, so put its label there.
            h, w = display.shape[:2]
            for arm, x in (("left", 20), ("right", w - 330)):
                color = {"UP": UP_COLOR, "DOWN": DOWN_COLOR}.get(states[arm], UNKNOWN_COLOR)
                cv2.putText(display, f"{arm.upper()} ARM: {states[arm]}", (x, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            if states["left"] == states["right"] == "UP":
                cv2.putText(display, "BOTH ARMS UP!", (w // 2 - 200, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, UP_COLOR, 4)
            if args.mirror_before_detect:
                cv2.putText(display, "mirrored BEFORE detection", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, display)
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