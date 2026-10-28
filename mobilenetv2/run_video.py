"""Run the drowsiness pipeline on a video file (no webcam, no Streamlit).

Same per-frame logic as video_frame_callback in app/streamlit_app.py:
FaceLandmarker -> EAR/MAR -> EMA -> CNN on face crop -> DrowsinessAlertSystem.

Usage:
    python run_video.py --video path/to/clip.mp4 --model MobileNetV2
    python run_video.py --video clip.mp4 --show
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402

from src.alert.alert_system import DrowsinessAlertSystem  # noqa: E402
from src.classification.predict import DrowsinessPredictor  # noqa: E402
from src.utils.drowsiness_utils import (  # noqa: E402
    compute_avg_ear,
    compute_drowsiness_score,
    compute_mar,
    drowsiness_level,
    extract_eye_landmarks,
    extract_mouth_landmarks,
)

EMA_ALPHA = 0.55  # same as the Streamlit app


class _LMAdapter:
    """extract_*_landmarks() expects the legacy `.landmark` interface."""

    __slots__ = ("landmark",)

    def __init__(self, lm_list):
        self.landmark = lm_list


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="input video path")
    p.add_argument("--model", default="MobileNetV2",
                   help="model name; loads models/weights/<model>_best.pt")
    p.add_argument("--outdir", default="video_results", help="output folder")
    p.add_argument("--ear", type=float, default=0.23, help="EAR threshold")
    p.add_argument("--mar", type=float, default=0.60, help="MAR threshold")
    p.add_argument("--consec", type=int, default=4,
                   help="consecutive drowsy frames before alert "
                        "(2 s rule = 2 * video FPS)")
    p.add_argument("--no_video", action="store_true",
                   help="skip writing the annotated video")
    p.add_argument("--show", action="store_true",
                   help="show a preview window (q to quit)")
    return p.parse_args()


def draw_overlay(frame, ear, mar, cnn, level, consec, t):
    color = {"ALERT": (80, 200, 80), "MILD": (0, 200, 255),
             "DROWSY": (0, 0, 255)}[level]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (30, 30, 30), -1)
    text = (f"{level}  EAR:{ear:.2f} MAR:{mar:.2f} CNN:{cnn:.2f} "
            f"C:{consec}  t={t:.1f}s")
    cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    args = parse_args()
    # the alert system logs a warning per drowsy/yawn frame; keep the console readable
    logging.getLogger("src.alert.alert_system").setLevel(logging.ERROR)

    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    task_path = ROOT / "models" / "face_landmarker.task"
    weights_path = ROOT / "models" / "weights" / f"{args.model}_best.pt"
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    predictor = DrowsinessPredictor(args.model, str(weights_path))
    asys = DrowsinessAlertSystem(
        ear_threshold=args.ear, mar_threshold=args.mar,
        consec_frames=args.consec, alert_sound=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{video_path.stem}_{args.model}.csv"
    writer = None
    if not args.no_video:
        out_video = outdir / f"{video_path.stem}_{args.model}.mp4"
        writer = cv2.VideoWriter(
            str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height))

    ema = {"ear": None, "mar": None}
    rows = []
    segments = []  # (start_s, end_s) while is_drowsy
    seg_start = None
    n_face = 0
    t_start = time.perf_counter()

    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = idx / fps
        h, w = img.shape[:2]

        proc_img = cv2.resize(img, None, fx=0.75, fy=0.75) if w > 480 else img
        ph, pw = proc_img.shape[:2]
        lab = cv2.cvtColor(proc_img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR),
                           cv2.COLOR_BGR2RGB)

        face = False
        raw_ear = raw_mar = None
        cnn_conf = 0.0
        res = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.face_landmarks:
            face = True
            n_face += 1
            lm = _LMAdapter(res.face_landmarks[0])
            le, re_ = extract_eye_landmarks(lm, pw, ph)
            raw_ear = compute_avg_ear(le, re_)
            raw_mar = compute_mar(extract_mouth_landmarks(lm, pw, ph))

            x_min = int(min(p.x for p in lm.landmark) * pw)
            y_min = int(min(p.y for p in lm.landmark) * ph)
            x_max = int(max(p.x for p in lm.landmark) * pw)
            y_max = int(max(p.y for p in lm.landmark) * ph)
            pad_x = int((x_max - x_min) * 0.15)
            pad_y = int((y_max - y_min) * 0.15)
            bbox = (max(0, x_min - pad_x), max(0, y_min - pad_y),
                    min(pw, x_max + pad_x), min(ph, y_max + pad_y))
            try:
                cnn_conf = predictor.predict_from_frame(
                    proc_img, bbox)["probabilities"]["drowsy"]
            except Exception:
                cnn_conf = 0.0

        if face:
            ema["ear"] = raw_ear if ema["ear"] is None else (
                EMA_ALPHA * raw_ear + (1 - EMA_ALPHA) * ema["ear"])
            ema["mar"] = raw_mar if ema["mar"] is None else (
                EMA_ALPHA * raw_mar + (1 - EMA_ALPHA) * ema["mar"])
            ear_v, mar_v = ema["ear"], ema["mar"]
        else:
            ema["ear"] = ema["mar"] = None
            ear_v, mar_v = 0.30, 0.40

        status = asys.update(ear=ear_v, mar=mar_v, cnn_confidence=cnn_conf)
        score = compute_drowsiness_score(ear=ear_v, mar=mar_v,
                                         cnn_confidence=cnn_conf)
        score_level = drowsiness_level(score)
        if status["is_drowsy"]:
            level = "DROWSY"
        elif status["is_yawning"] and score_level == "ALERT":
            level = "MILD"
        else:
            level = score_level

        if status["is_drowsy"] and seg_start is None:
            seg_start = t
        elif not status["is_drowsy"] and seg_start is not None:
            segments.append((seg_start, t))
            seg_start = None

        rows.append([idx, f"{t:.3f}", int(face), f"{ear_v:.4f}",
                     f"{mar_v:.4f}", f"{cnn_conf:.4f}", f"{score:.4f}",
                     level, status["consecutive_frames"],
                     int(status["is_drowsy"]), int(status["is_yawning"])])

        draw_overlay(img, ear_v, mar_v, cnn_conf, level,
                     status["consecutive_frames"], t)
        if writer is not None:
            writer.write(img)
        if args.show:
            cv2.imshow("run_video", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1
        if idx % 100 == 0:
            print(f"  {idx}/{total} frames", flush=True)

    if seg_start is not None:
        segments.append((seg_start, idx / fps))
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "face", "ear", "mar", "cnn_drowsy",
                     "score", "level", "consec", "is_drowsy", "is_yawning"])
        wr.writerows(rows)

    elapsed = time.perf_counter() - t_start
    n = max(idx, 1)
    n_drowsy = sum(r[9] for r in rows)
    print("=" * 50)
    print(f"video      : {video_path.name} ({width}x{height}, {fps:.1f} FPS, {idx} frames)")
    print(f"model      : {args.model}   ear<{args.ear} mar>{args.mar} consec={args.consec}")
    print(f"face found : {n_face}/{idx} frames ({100 * n_face / n:.1f}%)")
    print(f"drowsy     : {n_drowsy} frames ({100 * n_drowsy / n:.1f}%)")
    print(f"alerts     : {len(segments)} segment(s)")
    for s, e in segments:
        print(f"  {s:7.2f}s -> {e:7.2f}s  ({e - s:.2f}s)")
    print(f"processing : {idx / elapsed:.1f} FPS")
    print(f"csv        : {csv_path}")
    if writer is not None:
        print(f"video out  : {out_video}")
    print("=" * 50)


if __name__ == "__main__":
    main()
