"""
Lesson 06: Hand landmarks
=========================

What you'll learn
-----------------
- How every MediaPipe Tasks program is built:
    1. a MODEL file (.task) that holds the trained AI
    2. OPTIONS that say how to run it
    3. a DETECTOR made from the options
    4. call the detector on each frame and read the RESULT
- The three running modes: IMAGE, VIDEO and LIVE_STREAM
- The 21 hand landmarks and their numbers
- Normalized coordinates: x and y go from 0.0 to 1.0, not in pixels
- Handedness: is it a left or a right hand?

The 21 landmarks
----------------
    0  wrist
    1-4    thumb   (4 = thumb tip)
    5-8    index   (8 = index fingertip)
    9-12   middle  (12 = tip)
    13-16  ring    (16 = tip)
    17-20  pinky   (20 = tip)

Within each finger the numbers go from the knuckle out to the tip.

How to run
----------
    python lessons/part2_mediapipe/06_hand_landmarks.py
    python lessons/part2_mediapipe/06_hand_landmarks.py --hands 1 --min-confidence 0.7

Controls
--------
    n          show / hide landmark numbers
    p          print all 21 landmarks of the first hand to the terminal
    q or Esc   quit

Try this
--------
1. Press p and look at the z values. Which landmark is always 0? What does
   a negative z mean? (Move a finger toward the camera and press p again.)
2. Draw a big circle on the index fingertip (landmark 8) only.
3. Print the pixel distance between the thumb tip (4) and index tip (8).
   Pinch your fingers together: what's the smallest distance you get?
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

WINDOW_NAME = "Lesson 06 - Hand landmarks"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

# Short names for the MediaPipe classes we use.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Pairs of landmark numbers to connect with lines, to draw the hand "skeleton".
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                # index
    (9, 10), (10, 11), (11, 12),                   # middle
    (13, 14), (14, 15), (15, 16),                  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),         # pinky
    (5, 9), (9, 13), (13, 17),                     # across the palm
]
FINGERTIPS = [4, 8, 12, 16, 20]


def parse_args():
    parser = argparse.ArgumentParser(description="Find and draw hand landmarks with MediaPipe.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    parser.add_argument("--hands", type=int, default=2,
                        help="Maximum number of hands to look for. (default: 2)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="0-1. How sure the model must be before it reports a hand. (default: 0.5)")
    return parser.parse_args()


def create_detector(num_hands, min_confidence):
    """Build a HandLandmarker.

    Args:
        num_hands:      Maximum number of hands to detect.
        min_confidence: Minimum detection score (0-1).

    Returns:
        A ready-to-use HandLandmarker.

    About running_mode:
        IMAGE        one unrelated picture at a time (e.g. photos)
        VIDEO        frames from a video, in order. MediaPipe uses the previous
                     frame to find the hand faster. Needs a timestamp per frame.
        LIVE_STREAM  like VIDEO, but results arrive later through a callback
                     function, so the camera never waits for the model.
        We use VIDEO: it's fast and the code stays simple.
    """
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun:  python download_models.py")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_confidence,
    )
    return HandLandmarker.create_from_options(options)


def to_pixel(landmark, width, height):
    """Convert a normalized landmark (x, y from 0.0 to 1.0) to pixel coordinates.

    Args:
        landmark: A MediaPipe landmark with .x and .y attributes.
        width, height: Image size in pixels.

    Returns:
        (x, y) as whole numbers, ready for OpenCV drawing functions.
    """
    return int(landmark.x * width), int(landmark.y * height)


def draw_hand(frame, landmarks, show_numbers):
    """Draw one hand's skeleton.

    Args:
        frame:        Image to draw on (modified in place).
        landmarks:    List of 21 landmarks for one hand.
        show_numbers: Also write each landmark's number next to it.
    """
    h, w = frame.shape[:2]
    points = [to_pixel(lm, w, h) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
    for i, point in enumerate(points):
        color = (0, 0, 255) if i in FINGERTIPS else (0, 200, 0)
        cv2.circle(frame, point, 6 if i in FINGERTIPS else 4, color, -1)
        if show_numbers:
            cv2.putText(frame, str(i), (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def print_landmarks(landmarks):
    """Print a table of one hand's 21 landmarks."""
    print("\n  #       x       y       z")
    for i, lm in enumerate(landmarks):
        print(f"{i:3d}  {lm.x:6.3f}  {lm.y:6.3f}  {lm.z:6.3f}")
    print("x, y: 0 = left/top edge, 1 = right/bottom edge.")
    print("z: depth relative to the wrist (0). Smaller = closer to the camera.\n")


def main():
    args = parse_args()
    detector = create_detector(args.hands, args.min_confidence)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. See lesson 01 for help.")

    show_numbers = True
    start = time.monotonic()
    last_timestamp = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Mirror BEFORE detecting: MediaPipe's left/right hand labels assume
            # a mirrored, selfie-style picture.
            frame = cv2.flip(frame, 1)

            # MediaPipe wants RGB pixels in its own Image type. OpenCV gives BGR.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode needs each frame's time in milliseconds, and it must
            # always increase (MediaPipe stops with an error if two are equal).
            timestamp_ms = max(int((time.monotonic() - start) * 1000), last_timestamp + 1)
            last_timestamp = timestamp_ms
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # result.hand_landmarks: one list of 21 landmarks per hand found
            # result.handedness:     one list of guesses per hand, best guess first
            h, w = frame.shape[:2]
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                draw_hand(frame, landmarks, show_numbers)
                label = handedness[0].category_name          # "Left" or "Right"
                score = handedness[0].score
                wrist = to_pixel(landmarks[0], w, h)
                cv2.putText(frame, f"{label} ({score:.2f})", (wrist[0] - 40, wrist[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)

            cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}   n: numbers   p: print   q: quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("n"):
                show_numbers = not show_numbers
            if key == ord("p"):
                if result.hand_landmarks:
                    print_landmarks(result.hand_landmarks[0])
                else:
                    print("No hand in view.")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()