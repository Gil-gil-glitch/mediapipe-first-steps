"""
Lesson 02: Filters
==================

What you'll learn
-----------------
- Grayscale: turning 3 color channels into 1 brightness channel
- Blur: averaging each pixel with its neighbours (cv2.GaussianBlur)
- Edge detection: finding sudden changes in brightness (cv2.Canny)
- Threshold: turning every pixel pure black or pure white
- Invert: a filter you can write yourself with plain NumPy math

Most computer vision programs filter the picture first, to make the
important parts easier to find. For example, edge detection usually works
better on a blurred grayscale image than on the raw color picture.

How to run
----------
    python lessons/part1_opencv/02_filters.py
    python lessons/part1_opencv/02_filters.py --camera 1

Controls
--------
    1  original        4  edges
    2  grayscale       5  threshold
    3  blur            6  invert
    q or Esc  quit

Try this
--------
1. Change BLUR_SIZE to 5, then 51. Why must it be an odd number?
2. In edges(), change the Canny thresholds (50, 150). What happens with (10, 30)?
   With (200, 300)?
3. Add a 7th filter: press 7 to show only the red channel, frame[:, :, 2].
   (Remember to add it to the FILTERS dictionary.)
"""

import argparse

import cv2

WINDOW_NAME = "Lesson 02 - Filters"
BLUR_SIZE = 21          # size of the blur "window" in pixels: must be odd
THRESHOLD_LEVEL = 127   # 0-255: pixels brighter than this become white


# ---------------------------------------------------------------------------
# The filters. Each one takes a color frame and returns a new image.
# ---------------------------------------------------------------------------
def original(frame):
    """Return the frame unchanged."""
    return frame


def grayscale(frame):
    """Convert to one brightness channel. Note: OpenCV color order is BGR, not RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur(frame):
    """Gaussian blur: each pixel becomes a weighted average of its neighbours.

    The second argument is the size (width, height) of the neighbourhood.
    0 as the last argument lets OpenCV choose how strong the blur is.
    """
    return cv2.GaussianBlur(frame, (BLUR_SIZE, BLUR_SIZE), 0)


def edges(frame):
    """Canny edge detection: white where brightness changes sharply.

    The two numbers are thresholds. Changes stronger than 150 are definitely
    edges; changes between 50 and 150 count only if they touch a strong edge.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)      # blur first to remove noise
    return cv2.Canny(gray, 50, 150)


def threshold(frame):
    """Every pixel becomes black (0) or white (255).

    cv2.threshold returns two things: the threshold it used, and the image.
    We only need the image, so the first value goes into "_".
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_LEVEL, 255, cv2.THRESH_BINARY)
    return binary


def invert(frame):
    """A filter in one line of NumPy: every pixel value v becomes 255 - v."""
    return 255 - frame


# Key -> (name shown on screen, filter function)
FILTERS = {
    "1": ("Original", original),
    "2": ("Grayscale", grayscale),
    "3": ("Blur", blur),
    "4": ("Edges", edges),
    "5": ("Threshold", threshold),
    "6": ("Invert", invert),
}


def add_label(image, text):
    """Write the filter name in the corner.

    Grayscale images have only one channel, so we convert them back to
    3 channels first; otherwise we couldn't draw colored text on them.

    Args:
        image: The filtered image (1 or 3 channels).
        text:  Label to write.

    Returns:
        A 3-channel image with the label drawn on it.
    """
    if image.ndim == 2:                       # 2 dimensions = height x width = grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(image, "keys 1-6: change filter   q: quit", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Apply filters to the webcam feed.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera number. (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. See lesson 01 for help.")

    current = "1"     # start with the original picture
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            name, filter_function = FILTERS[current]
            output = filter_function(frame)
            cv2.imshow(WINDOW_NAME, add_label(output, name))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if chr(key) in FILTERS:          # chr turns a key code back into a character
                current = chr(key)
                print(f"Filter: {FILTERS[current][0]}")
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()