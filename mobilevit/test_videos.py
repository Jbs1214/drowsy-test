"""폴더 안의 mp4를 화면 없이 판정하고 영상별 결과를 CSV로 저장한다.

python test_videos.py ../sleep results.csv
"""
import csv
import glob
import os
import sys
from collections import deque

import cv2

from app import THRESHOLD, WINDOW_SEC, crop_largest_face
from predict import load_model, predict_drowsy

FRAME_STEP = 3  # 30fps 영상을 10fps로 샘플링 (웹캠 실사용 FPS에 가깝게)


def test_video(model, path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    history = deque()
    probs, drowsy_frames, first_alert, n, i = [], 0, None, 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % FRAME_STEP:
            continue
        n += 1
        t = i / fps
        face, _ = crop_largest_face(frame)
        if face is None:
            history.clear()
            continue
        prob = predict_drowsy(model, face)
        probs.append(prob)
        history.append((t, prob))
        while t - history[0][0] > WINDOW_SEC:
            history.popleft()
        full = t - history[0][0] >= WINDOW_SEC * 0.9
        avg = sum(p for _, p in history) / len(history)
        if full and avg > THRESHOLD:
            drowsy_frames += 1
            if first_alert is None:
                first_alert = t
    cap.release()
    return {
        "video": os.path.basename(path),
        "duration_sec": round((i + 1) / fps, 1),
        "face_ratio": round(len(probs) / n, 3) if n else 0,
        "mean_prob": round(sum(probs) / len(probs), 3) if probs else "",
        "drowsy_ratio": round(drowsy_frames / n, 3) if n else 0,
        "alerted": first_alert is not None,
        "first_alert_sec": round(first_alert, 1) if first_alert is not None else "",
    }


def main():
    folder, out = sys.argv[1], sys.argv[2]
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.mp4"), recursive=True))
    paths = list({os.path.basename(p): p for p in paths}.values())  # 같은 이름의 중복 영상 제외
    model = load_model()
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = None
        for k, path in enumerate(paths, 1):
            row = test_video(model, path)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f"[{k}/{len(paths)}] {row}", flush=True)


if __name__ == "__main__":
    main()
