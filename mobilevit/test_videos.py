"""폴더 안의 mp4를 화면 없이 판정하고 영상별 결과를 CSV로 저장한다.

python test_videos.py ../sleep results.csv
"""
import csv
import glob
import os
import sys

import cv2

from detector import DrowsyDetector
from predict import load_model

FRAME_STEP = 3  # 30fps 영상을 10fps로 샘플링 (웹캠 실사용 FPS에 가깝게)
REASONS = ["MODEL", "EYES", "YAWN", "NOD"]


def test_video(model, path):
    detector = DrowsyDetector(model)  # 영상마다 시간이 0부터 시작하므로 새로 만든다
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    probs, drowsy_frames, first_alert, n, i = [], 0, None, 0, -1
    first = {r: None for r in REASONS}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % FRAME_STEP:
            continue
        n += 1
        t = i / fps
        box, prob, _, reasons = detector.update(frame, t)
        if box is None:
            continue
        probs.append(prob)
        if reasons:
            drowsy_frames += 1
            if first_alert is None:
                first_alert = t
        for r in reasons:
            if first[r] is None:
                first[r] = t
    cap.release()
    detector.close()
    row = {
        "video": os.path.basename(path),
        "duration_sec": round((i + 1) / fps, 1),
        "face_ratio": round(len(probs) / n, 3) if n else 0,
        "mean_prob": round(sum(probs) / len(probs), 3) if probs else "",
        "drowsy_ratio": round(drowsy_frames / n, 3) if n else 0,
        "alerted": first_alert is not None,
        "first_alert_sec": round(first_alert, 1) if first_alert is not None else "",
    }
    # 이유별 첫 경고 시각(초). 비어 있으면 그 이유로는 경고가 없었다.
    for r in REASONS:
        row[r.lower()] = round(first[r], 1) if first[r] is not None else ""
    return row


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
