import time
import winsound
from tkinter import Tk, filedialog

import cv2

from detector import DrowsyDetector

BEEP_INTERVAL = 1.0


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
    detector = DrowsyDetector()
    path = choose_video()
    cap = cv2.VideoCapture(path if path else 0)
    start = time.time()
    last_beep = 0.0
    prev = start

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        # 영상은 추론 때문에 느리게 재생되므로 판정 시간은 영상 시간 기준
        t = cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FPS) if path else now - start

        box, reasons = detector.update(frame, t)
        drowsy = bool(reasons)
        if box is None:
            cv2.putText(frame, "No face", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            color = (0, 0, 255) if drowsy else (0, 255, 0)
            cv2.rectangle(frame, box[:2], box[2:], color, 2)

        if drowsy:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 20)
            cv2.putText(frame, "DROWSY! WAKE UP", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            cv2.putText(frame, " ".join(reasons), (40, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if now - last_beep >= BEEP_INTERVAL:
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                last_beep = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Drowsiness Detection (q: quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
