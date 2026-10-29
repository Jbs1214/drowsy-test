import time
import winsound
from collections import deque
from tkinter import Tk, filedialog

import cv2
from PIL import Image

from predict import load_model, predict_drowsy

WINDOW_SEC = 2.0   # 판정 구간
THRESHOLD = 0.7    # 구간 평균 Drowsy 확률 기준
BEEP_INTERVAL = 1.0
MARGIN = 0.2       # 얼굴 박스 여유 비율

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_largest_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * MARGIN), int(h * MARGIN)
    x1, y1 = max(x - mx, 0), max(y - my, 0)
    x2, y2 = min(x + w + mx, frame.shape[1]), min(y + h + my, frame.shape[0])
    face = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    return face, (x1, y1, x2, y2)


def choose_video():
    """영상 파일 경로를 반환한다. 취소하면 빈 문자열(웹캠 사용)."""
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="영상 선택 (취소하면 웹캠)",
        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def main():
    model = load_model()
    path = choose_video()
    cap = cv2.VideoCapture(path if path else 0)
    history = deque()  # (time, prob)
    last_beep = 0.0
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        # 영상은 추론 때문에 느리게 재생되므로 판정 구간은 영상 시간 기준
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if path else now

        face, box = crop_largest_face(frame)
        drowsy = False
        if face is None:
            history.clear()
            cv2.putText(frame, "No face", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            prob = predict_drowsy(model, face)
            history.append((t, prob))
            while t - history[0][0] > WINDOW_SEC:
                history.popleft()
            # 구간이 2초를 거의 다 채웠을 때만 판정 (프레임 간격 오차 허용)
            full = t - history[0][0] >= WINDOW_SEC * 0.9
            avg = sum(p for _, p in history) / len(history)
            drowsy = full and avg > THRESHOLD

            color = (0, 0, 255) if drowsy else (0, 255, 0)
            cv2.rectangle(frame, box[:2], box[2:], color, 2)
            cv2.putText(frame, f"Drowsy: {prob:.2f} (avg {avg:.2f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if drowsy:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 20)
            cv2.putText(frame, "DROWSY! WAKE UP", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            if now - last_beep >= BEEP_INTERVAL:
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                last_beep = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Drowsiness Detection (q: quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
