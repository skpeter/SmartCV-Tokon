"""Replay a Tokon VOD at SmartCV's poll interval and print detector events.

Usage (from repo root):
    python dev/validate_vod.py
    python dev/validate_vod.py path/to.mp4 --start 50 --end 380 --step 0.5
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if not os.path.exists("config.ini"):
    shutil.copy(os.path.join("core", "config.ini.example"), "config.ini")

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

import routines  # noqa: E402


DEFAULT_VOD = r"D:\vod_library\YTDown.com_YouTube_Media_lKjh_y4I8W0_001_1080p.mp4"


def snapshot(payload):
    p = payload["players"]
    return (
        payload["state"],
        payload["round"],
        p[0]["rounds"],
        p[1]["rounds"],
        p[0]["games"],
        p[1]["games"],
        p[0]["character"],
        p[1]["character"],
    )


def fmt(payload):
    p = payload["players"]
    chars = f"{p[0]['character'] or '-'} vs {p[1]['character'] or '-'}"
    return (
        f"state={payload['state']} round={payload['round']} "
        f"rounds={p[0]['rounds']}-{p[1]['rounds']} "
        f"games={p[0]['games']}-{p[1]['games']} {chars}"
    )


def run(path, start, end, step, ocr):
    routines.ocr_enabled = ocr
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")

    t = start
    prev = None
    print(f"scan {path}")
    print(f"range {start}s .. {end}s step {step}s ocr={ocr}")
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        scale_x = w / 1920
        scale_y = h / 1080
        routines._now = lambda ts=t: ts
        funcs = routines.states_to_functions.get(routines.payload.get("state"), [])
        for func in funcs:
            func(routines.payload, img, scale_x, scale_y)
        cur = snapshot(routines.payload)
        if cur != prev:
            print(f"  t={t:7.1f}  {fmt(routines.payload)}")
            prev = cur
        t += step
    cap.release()
    print("done", fmt(routines.payload))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("vod", nargs="?", default=DEFAULT_VOD)
    p.add_argument("--start", type=float, default=50.0)
    p.add_argument("--end", type=float, default=380.0)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--ocr", action="store_true", help="run one-shot leader OCR")
    args = p.parse_args()
    run(args.vod, args.start, args.end, args.step, args.ocr)


if __name__ == "__main__":
    main()
