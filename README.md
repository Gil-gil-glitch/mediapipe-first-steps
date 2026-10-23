# MediaPipe First Steps

Short, hands-on Python lessons for computer vision with **OpenCV** and
**MediaPipe**: from opening a webcam to tracking hands and body poses and
recording the data.

Each lesson is **one self-contained Python file**. Run it, read it, break it,
change it. Every file starts with a docstring explaining what you'll learn,
how to run it, the keyboard controls, and a few "Try this" exercises.

> **A note on old tutorials:** most MediaPipe videos and blog posts online use
> `mp.solutions` (for example `mp.solutions.hands`). That older API has been
> removed from recent MediaPipe versions, so that code no longer runs.
> These lessons use the current **MediaPipe Tasks** API.

## Setup (macOS)

You need Python 3.10, 3.11 or 3.12. Check with `python3 --version`.

```bash
# 1. Get the code
git clone <this repository's URL>
cd mediapipe-first-steps

# 2. Create a virtual environment (a private set of packages for this project)
python3 -m venv .venv
source .venv/bin/activate          # run this every time you open a new terminal

# 3. Install the packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download the MediaPipe models (needed for Part 2)
python download_models.py
```

**Camera permission:** the first time a lesson opens the camera, macOS asks
whether your terminal app (Terminal, iTerm, VS Code...) may use the camera.
Click **Allow**. If you clicked "Don't Allow" by accident, go to
**System Settings → Privacy & Security → Camera**, switch your terminal app
on, then quit and reopen the terminal.

**Wrong camera?** If you have an iPhone nearby, macOS may pick it as camera 0
(Continuity Camera). Use the built-in camera with `--camera 1`.

## Running a lesson

From the repository folder, with the virtual environment active:

```bash
python lessons/part1_opencv/open_camera.py
python lessons/part1_opencv/open_camera.py --help    # see all options
```

Press **q** (or Esc) in the video window to quit.
Anything a lesson saves (videos, snapshots, CSV files) goes into `output/`.

## Lessons

### Part 1: OpenCV basics

| # | Lesson | You'll learn |
|---|---|---|
| 01 | `01_open_camera.py` | Open a webcam, the read/show loop, mirroring, quitting cleanly |
| 02 | `02_filters.py` | Grayscale, blur, edge detection, threshold; switch filters with keys |
| 03 | `03_drawing_and_text.py` | Lines, boxes, circles, text, an FPS counter, mouse clicks, BGR colors |
| 04 | `04_recording.py` | Record video and take snapshots |
| 05 | `05_color_tracking.py` | HSV color, masks, contours: track a colored object |

### Part 2: MediaPipe

| # | Lesson | You'll learn |
|---|---|---|
| 06 | `06_hand_landmarks.py` | Model files, running modes, the 21 hand points |
| 07 | `07_finger_counting.py` | Count raised fingers with your own geometry rules |
| 08 | `08_gesture_recognizer.py` | MediaPipe's 7 built-in gestures and confidence scores |
| 09 | `09_pose_landmarks.py` | The 33 body points and their visibility scores |
| 10 | `10_arm_raised.py` | Detect which arm is raised |
| 11 | `11_joint_angles.py` | Measure elbow and shoulder angles from three points |
| 12 | `12_record_pose_data.py` | Save pose data to a CSV file, and what each value means |
| 13 | `13_plot_pose_data.py` | Plot recorded data (no camera needed) |

### Assignment

See [`assignment/ASSIGNMENT.md`](assignment/ASSIGNMENT.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera 0` | Allow camera access (see Setup), close other apps using the camera (Zoom, FaceTime), or try `--camera 1` |
| Black window or no window on macOS | Check camera permission; quit and reopen the terminal after changing it |
| `ModuleNotFoundError: No module named 'cv2'` | Activate the virtual environment: `source .venv/bin/activate` |
| `Model not found` | Run `python download_models.py` |
| SSL / certificate error when downloading models | Finder → Applications → Python 3.x → double-click **Install Certificates.command** |
| Recorded video won't play | Open it with QuickTime or VLC; make sure you stopped recording (press `r` again) before quitting |
| Video plays too fast or too slow | See the `--fps` option in lesson 04 |