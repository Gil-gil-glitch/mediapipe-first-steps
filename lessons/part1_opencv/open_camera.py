"""
Lesson 01: Open the camera
==========================

What you'll learn
-----------------
- How to open a webcam with OpenCV (cv2.VideoCapture)
- The "read a frame, show a frame" loop that every live video program uses
- What a frame actually is: a NumPy array of pixels
- How to mirror the picture and quit cleanly

How to run
----------
    python lessons/part1_opencv/01_open_camera.py
    python lessons/part1_opencv/01_open_camera.py --camera 1
    python lessons/part1_opencv/01_open_camera.py --width 640 --height 480 --no-mirror

Controls
--------
    q or Esc   quit (closing the window also works)

Try this
--------
1. Print frame.shape inside the loop. What do the three numbers mean?
2. Print frame[0, 0]: the color of the top-left pixel. Point the camera at
   something red. Why is the biggest number the LAST one? (Hint: lesson 03.)
3. Change cv2.flip(frame, 1) to cv2.flip(frame, 0), then to -1. What happens?
"""

import argparse

import cv2

WINDOW_NAME = "Lesson 01 - Camera"


def parse_args():
    """Read the command-line options. Run with --help to see them all."""
    parser = argparse.ArgumentParser(description="Show the webcam feed in a window.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera number. 0 is usually the built-in camera. (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Requested frame width in pixels. (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Requested frame height in pixels. (default: 720)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Show the picture as the camera sees it instead of like a mirror.")
    return parser.parse_args()


def open_camera(index, width, height):
    """Open a camera and ask for a frame size.

    Args:
        index:  Camera number (0, 1, 2, ...).
        width:  Requested width in pixels.
        height: Requested height in pixels.

    Returns:
        An opened cv2.VideoCapture object.

    Raises:
        SystemExit: If the camera can't be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {index}.\n"
            "  - macOS: allow camera access for your terminal in\n"
            "    System Settings -> Privacy & Security -> Camera\n"
            "  - Close other apps using the camera (Zoom, FaceTime, ...)\n"
            "  - Try another camera number, e.g. --camera 1")

    # A camera is only *asked* for this size. It picks the closest size it supports.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def window_closed(name):
    """Return True if the user closed the window with its close button."""
    return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1


def main():
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height)

    # Read one frame to find out what size the camera actually gave us.
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("The camera opened but sent no picture. Check camera permission.")
    height, width, channels = frame.shape
    print(f"Camera {args.camera}: {width} x {height} pixels, {channels} color channels")
    print("Press q in the video window to quit.")

    cv2.namedWindow(WINDOW_NAME)
    try:
        while True:
            # 1. Grab the next picture from the camera.
            #    ok is False if no picture arrived (camera unplugged, etc.).
            ok, frame = cap.read()
            if not ok:
                print("No frame received, stopping.")
                break

            # 2. Mirror it (flip left-right) so moving your hand left moves it left on screen.
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            # 3. Show it.
            cv2.imshow(WINDOW_NAME, frame)

            # 4. Wait 1 ms for a key press. This ALSO gives the window time to
            #    draw itself: without waitKey, imshow shows nothing.
            #    "& 0xFF" keeps only the last 8 bits so key codes compare correctly.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # 27 is the Esc key
                break
            if window_closed(WINDOW_NAME):
                break
    finally:
        # Always give the camera back, even if the program crashes,
        # otherwise other apps (or the next run) may not be able to use it.
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)   # macOS needs one more waitKey to actually close the window


if __name__ == "__main__":
    main()