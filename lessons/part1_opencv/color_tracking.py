"""
Lesson 03: Drawing and text
===========================

What you'll learn
-----------------
- Drawing lines, rectangles and circles on a live frame
- Writing text, including a live FPS (frames per second) counter
- OpenCV colors are (Blue, Green, Red), NOT (Red, Green, Blue)
- Pixel coordinates: (0, 0) is the TOP-LEFT corner and y grows DOWNWARD
- Reacting to mouse clicks with a callback function
- Semi-transparent overlays with cv2.addWeighted

Everything here comes back in Part 2: MediaPipe gives you points, and you
draw them with exactly these functions.

How to run
----------
    python lessons/part1_opencv/03_drawing_and_text.py

Controls
--------
    left click  drop a marker where you clicked
    c           clear all markers
    q or Esc    quit

Try this
--------
1. Change RED to (255, 0, 0). What color are the markers now? Why?
2. Make the crosshair follow your mouse instead of staying in the centre.
   (Hint: the mouse callback also receives cv2.EVENT_MOUSEMOVE events.)
3. Draw a line connecting the markers in the order you clicked them.
"""

import argparse
import time
from datetime import datetime

import cv2

WINDOW_NAME = "Lesson 03 - Drawing"

# Colors are (Blue, Green, Red), each 0-255.
RED = (0, 0, 255)
GREEN = (0, 200, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class FpsCounter:
    """Measures frames per second, smoothed so the number doesn't jump around.

    Usage:
        fps = FpsCounter()
        while True:
            ...
            fps.tick()          # call once per frame
            print(fps.value)
    """

    def __init__(self, smoothing=0.9):
        """
        Args:
            smoothing: 0-1. Higher = steadier number but slower to react.
        """
        self.smoothing = smoothing
        self.value = 0.0
        self._last = time.perf_counter()

    def tick(self):
        """Call once per frame to update self.value."""
        now = time.perf_counter()
        instant = 1.0 / max(now - self._last, 1e-6)   # avoid dividing by zero
        self._last = now
        # Exponential moving average: mostly the old value, a little of the new one.
        if self.value == 0.0:
            self.value = instant
        else:
            self.value = self.smoothing * self.value + (1 - self.smoothing) * instant


def on_mouse(event, x, y, flags, markers):
    """Called by OpenCV whenever the mouse does something in our window.

    We register this function with cv2.setMouseCallback below; we never call
    it ourselves. This is called a "callback".

    Args:
        event:   What happened, e.g. cv2.EVENT_LBUTTONDOWN (left click).
        x, y:    Mouse position in pixels.
        flags:   Extra info (Shift/Ctrl held, etc.). Unused here.
        markers: The list we passed as the last argument to setMouseCallback.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        markers.append((x, y))
        print(f"Marker at x={x}, y={y}")


def draw_info_box(frame, lines):
    """Draw text lines on a semi-transparent black box in the top-left corner.

    Args:
        frame: Image to draw on (modified in place).
        lines: List of strings, one per line.
    """
    box_height = 20 + 30 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, box_height), BLACK, thickness=-1)  # -1 = filled
    # Blend: 60% overlay + 40% original = see-through box.
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 38 + 30 * i), FONT, 0.65, WHITE, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw shapes and text on the webcam feed.")
    parser.add_argument("--camera", type=int, default=0, help="Camera number. (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. See lesson 01 for help.")

    markers = []                       # filled in by on_mouse
    fps = FpsCounter()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, markers)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2    # "//" divides and rounds down: pixels must be whole numbers

            # Crosshair in the centre: two lines and a circle.
            cv2.line(frame, (cx - 40, cy), (cx + 40, cy), GREEN, 2)
            cv2.line(frame, (cx, cy - 40), (cx, cy + 40), GREEN, 2)
            cv2.circle(frame, (cx, cy), 25, GREEN, 2)

            # Rectangle: top-left corner and bottom-right corner.
            cv2.rectangle(frame, (w - 220, h - 160), (w - 20, h - 20), YELLOW, 3)
            cv2.putText(frame, "a box", (w - 205, h - 120), FONT, 0.8, YELLOW, 2)

            # Markers from mouse clicks: filled red circles with their coordinates.
            for (x, y) in markers:
                cv2.circle(frame, (x, y), 8, RED, -1)
                cv2.putText(frame, f"({x},{y})", (x + 12, y - 12), FONT, 0.5, RED, 2)

            fps.tick()
            draw_info_box(frame, [
                f"FPS: {fps.value:5.1f}",
                f"Size: {w} x {h}",
                datetime.now().strftime("Time: %H:%M:%S"),
                f"Markers: {len(markers)} (c = clear)",
            ])

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                markers.clear()
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()