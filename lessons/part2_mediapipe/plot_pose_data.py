"""
Lesson 13: Plotting pose data
=============================

What you'll learn
-----------------
- Reading a CSV file with Python's built-in csv module
- Plotting values over time with matplotlib
- Calculating NEW information (an elbow angle) from recorded data,
  long after the camera is off

No camera needed: this lesson only reads a file made by lesson 12.

How to run
----------
    python lessons/part2_mediapipe/13_plot_pose_data.py                   # newest recording
    python lessons/part2_mediapipe/13_plot_pose_data.py output/pose_2026-10-23_14-05-09.csv
    python lessons/part2_mediapipe/13_plot_pose_data.py --landmark left_wrist --no-show

The plot is shown in a window and also saved next to the CSV as a .png file.

Try this
--------
1. Record yourself doing 5 slow arm curls with lesson 12, then plot it.
   Can you count the curls from the elbow angle line?
2. Plot the wrist's world height (wy) instead of the picture height (y).
   Do they have the same shape? Why is one upside down?
3. Find the frame where the elbow angle was smallest and print its time.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def newest_csv():
    """Return the most recently created pose_*.csv in output/, or None."""
    files = sorted(OUTPUT_DIR.glob("pose_*.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_csv(path):
    """Read a CSV file made by lesson 12.

    Args:
        path: Path to the .csv file.

    Returns:
        A dict {column name: list of floats}, e.g. data["right_wrist_y"][0]
        is the right wrist's y value in the first row.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)          # each row becomes a dict {column: text}
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name, text in row.items():
                data[name].append(float(text))
    return data


def angle_at(a, b, c):
    """Angle at point b in degrees between b->a and b->c (same as lesson 11)."""
    ba = [p - q for p, q in zip(a, b)]
    bc = [p - q for p, q in zip(c, b)]
    length = math.dist(a, b) * math.dist(c, b)
    if length == 0:
        return float("nan")                  # "not a number": matplotlib leaves a gap
    cosine = sum(p * q for p, q in zip(ba, bc)) / length
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def elbow_angles(data, side):
    """Compute the elbow angle for every row, using the 3D world coordinates.

    Args:
        data: Output of load_csv().
        side: "left" or "right".

    Returns:
        A list of angles in degrees, one per row.
    """
    def point(name, row):
        return (data[f"{side}_{name}_wx"][row], data[f"{side}_{name}_wy"][row],
                data[f"{side}_{name}_wz"][row])

    return [angle_at(point("shoulder", r), point("elbow", r), point("wrist", r))
            for r in range(len(data["time_s"]))]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pose data recorded by lesson 12.")
    parser.add_argument("csv", nargs="?", type=Path,
                        help="CSV file to plot. (default: the newest one in output/)")
    parser.add_argument("--landmark", default="right_wrist",
                        help="Landmark whose height to plot. (default: right_wrist)")
    parser.add_argument("--no-show", action="store_true",
                        help="Only save the .png, don't open a window.")
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.csv or newest_csv()
    if path is None or not path.exists():
        sys.exit("No CSV file found. Record one with lesson 12 first.")
    print(f"Reading {path}")

    data = load_csv(path)
    time_s = data["time_s"]
    if not time_s:
        sys.exit("The file has no rows. Was a person in view while recording?")
    column = f"{args.landmark}_y"
    if column not in data:
        sys.exit(f"No column '{column}'. Landmarks in this file: "
                 f"{sorted({name.rsplit('_', 1)[0] for name in data if name.endswith('_x')})}")
    print(f"{len(time_s)} rows, {time_s[-1]:.1f} seconds")

    # Two plots stacked on top of each other, sharing the time axis.
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    # y grows downward in the picture, so 1 - y turns it into "height".
    top.plot(time_s, [1 - y for y in data[column]], label=args.landmark)
    top.set_ylabel("height in picture (0-1)")
    top.set_title(f"{args.landmark} height")
    top.grid(alpha=0.3)

    for side in ("left", "right"):
        if f"{side}_elbow_wx" in data:
            bottom.plot(time_s, elbow_angles(data, side), label=f"{side} elbow")
    bottom.set_ylabel("angle (degrees)")
    bottom.set_xlabel("time (s)")
    bottom.set_title("Elbow angles (3D, from world landmarks)")
    bottom.set_ylim(0, 190)
    bottom.axhline(180, color="grey", linestyle=":", linewidth=1)   # straight arm
    bottom.legend()
    bottom.grid(alpha=0.3)

    fig.suptitle(path.name)
    fig.tight_layout()
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"Saved {png}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()