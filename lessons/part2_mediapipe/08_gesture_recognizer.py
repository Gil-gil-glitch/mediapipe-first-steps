"""
Lesson 08: Gesture recognizer
=============================

What you'll learn
-----------------
- Using a model that already knows some gestures, instead of writing rules
- Confidence scores: the model's guess for EVERY gesture, not just the winner
- "Debouncing": only trusting a gesture after it has been held for a few frames
- Connecting a gesture to an action

MediaPipe's built-in gestures
-----------------------------
    None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up,
    Victory, ILoveYou

"None" means a hand was found but it isn't one of the seven gestures.

Rules (lesson 07) vs a trained model (this lesson)
--------------------------------------------------
Rules are easy to understand and change, but you must think of every case.
A trained model handles messy real hands well, but you can't easily change
what it recognises. (You *can* train your own gestures with a tool called
MediaPipe Model Maker, but that's a project for later.)

How to run
----------
    python lessons/part2_mediapipe/08_gesture_recognizer.py
    python lessons/part2_mediapipe/08_gesture_recognizer.py --hold 15

Controls
--------
    q or Esc   quit

Try this
--------
1. Run with --hold 1, then --hold 30. Why is 1 annoying in a real app?
2. Change ACTIONS so Thumb_Down prints "Boo!" instead.
3. Which two gestures does the model confuse most often? Watch the bars.
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Lesson 08 - Gesture recognizer"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
ClassifierOptions = mp.tasks.components.processors.ClassifierOptions
RunningMode = mp.tasks.vision.RunningMode

GESTURES = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up",
            "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]

# What to do when a gesture is confirmed. Change these!
ACTIONS = {
    "Thumb_Up":    "Nice!",
    "Thumb_Down":  "Oh no...",
    "Victory":     "Peace!",
    "Open_Palm":   "Hello!",
    "Closed_Fist": "Grab!",
    "Pointing_Up": "Look up!",
    "ILoveYou":    "Love it!",
}


class VideoClock:
    """Always-increasing timestamps in milliseconds for VIDEO mode (see lesson 06)."""

    def __init__(self):
        self.start = time.monotonic()
        self.last = -1

    def now_ms(self):
        self.last = max(int((time.monotonic() - self.start) * 1000), self.last + 1)
        return self.last


class GestureDebouncer:
    """Confirms a gesture only after it has been seen `hold` frames in a row.

    Without this, a gesture that flickers for a single frame while your hand
    changes shape would trigger an action.
    """

    def __init__(self, hold):
        """
        Args:
            hold: Number of identical frames needed to confirm a gesture.
        """
        self.hold = hold
        self.candidate = None     # gesture we're currently watching
        self.count = 0            # how many frames in a row we've seen it
        self.confirmed = None     # the last confirmed gesture

    def update(self, gesture):
        """Feed in this frame's gesture.

        Returns:
            The gesture name if it has JUST been confirmed, otherwise None.
        """
        if gesture == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = gesture, 1

        if self.count >= self.hold and gesture != self.confirmed:
            self.confirmed = gesture
            return gesture
        return None

    @property
    def progress(self):
        """0.0-1.0: how close the current candidate is to being confirmed."""
        return min(self.count / self.hold, 1.0)


def draw_score_bars(frame, scores, x=20, y=70):
    """Draw one horizontal bar per gesture showing the model's confidence.

    Args:
        frame:  Image to draw on (modified in place).
        scores: Dict {gesture name: score 0-1}.
        x, y:   Top-left position of the bar chart.
    """
    best = max(scores, key=scores.get) if scores else None
    for i, name in enumerate(GESTURES):
        score = scores.get(name, 0.0)
        top = y + i * 30
        color = (0, 220, 0) if name == best else (200, 200, 200)
        cv2.rectangle(frame, (x + 140, top), (x + 140 + 200, top + 20), (60, 60, 60), -1)
        cv2.rectangle(frame, (x + 140, top), (x + 140 + int(200 * score), top + 20), color, -1)
        cv2.putText(frame, name, (x, top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"{score:.2f}", (x + 350, top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Recognise hand gestures with MediaPipe.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--hold", type=int, default=8,
                        help="Frames a gesture must be held before it counts. (default: 8)")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="Ignore gestures with a lower score than this. (default: 0.5)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun:  python download_models.py")

    recognizer = GestureRecognizer.create_from_options(GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        # Ask for the score of every gesture, not only the best one.
        canned_gesture_classifier_options=ClassifierOptions(max_results=len(GESTURES)),
    ))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. See lesson 01 for help.")

    clock = VideoClock()
    debouncer = GestureDebouncer(args.hold)
    message, message_until = "", 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), clock.now_ms())

            # result.gestures has one list per hand; each list holds Category
            # objects (category_name, score), best first.
            scores = {}
            gesture = "None"
            if result.gestures:
                scores = {c.category_name: c.score for c in result.gestures[0]}
                top = result.gestures[0][0]
                if top.score >= args.min_score:
                    gesture = top.category_name

            confirmed = debouncer.update(gesture)
            if confirmed in ACTIONS:
                message, message_until = ACTIONS[confirmed], time.monotonic() + 2.0
                print(f"{confirmed} -> {ACTIONS[confirmed]}")

            # Hand landmarks come free with the gesture result: mark the fingertips.
            h, w = frame.shape[:2]
            for hand in result.hand_landmarks:
                for i in (4, 8, 12, 16, 20):
                    cv2.circle(frame, (int(hand[i].x * w), int(hand[i].y * h)), 6, (0, 0, 255), -1)

            draw_score_bars(frame, scores)
            # Progress bar under the chart: how long the gesture has been held.
            bar_y = 70 + len(GESTURES) * 30 + 10
            cv2.putText(frame, f"holding: {debouncer.candidate}", (20, bar_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (160, bar_y), (160 + int(200 * debouncer.progress), bar_y + 20),
                          (0, 200, 255), -1)

            if time.monotonic() < message_until:
                cv2.putText(frame, message, (w // 2 - 120, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 255), 5)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()