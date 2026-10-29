import math
import os
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")
LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

EYE_CLOSED = 0.5      # 눈: eyeBlink 점수(0~1)가 이보다 크면 감은 것으로 본다
EYE_CLOSED_SEC = 1.0  # 눈: 이 시간 이상 계속 감으면 졸음
YAWN_OPEN = 0.5       # 하품: jawOpen 점수(0~1)가 이보다 크면 입을 벌린 것으로 본다
YAWN_SEC = 1.0        # 하품: 이 시간 이상 계속 벌리면 하품
NOD_DEG = 15.0        # 끄덕임: 고개가 이 각도 이상 숙여지면 끄덕임
NOD_SEC = 2.0         # 끄덕임: 최근 이 시간 중 가장 든 자세와 비교


def load_landmarker():
    if not os.path.exists(LANDMARKER_PATH):
        os.makedirs(MODELS_DIR, exist_ok=True)
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


class DrowsyDetector:
    """프레임마다 update()를 부르면 졸음 이유 목록을 돌려준다.
    이유: EYES(눈 감음), YAWN(하품), NOD(고개 끄덕임)
    """

    def __init__(self):
        self.landmarker = load_landmarker()
        self.pitches = deque()  # (t, 고개 숙임 각도)
        self.eye_start = None   # 눈을 감기 시작한 시각
        self.yawn_start = None  # 입을 벌리기 시작한 시각

    def close(self):
        self.landmarker.close()

    def update(self, frame, t):
        """t는 초 단위이고 계속 증가해야 한다. (box, reasons)를 반환하고, 얼굴이 없으면 box는 None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
        if not result.face_landmarks:
            self.pitches.clear()
            self.eye_start = self.yawn_start = None
            return None, []

        h, w = frame.shape[:2]
        xs = [p.x * w for p in result.face_landmarks[0]]
        ys = [p.y * h for p in result.face_landmarks[0]]
        box = (max(int(min(xs)), 0), max(int(min(ys)), 0), min(int(max(xs)), w), min(int(max(ys)), h))
        reasons = []

        # 눈 감음·하품: 점수가 기준을 넘은 상태가 일정 시간 이어지면
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        eye = (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2
        if eye <= EYE_CLOSED:
            self.eye_start = None
        elif self.eye_start is None:
            self.eye_start = t
        if self.eye_start is not None and t - self.eye_start >= EYE_CLOSED_SEC:
            reasons.append("EYES")
        if scores["jawOpen"] <= YAWN_OPEN:
            self.yawn_start = None
        elif self.yawn_start is None:
            self.yawn_start = t
        if self.yawn_start is not None and t - self.yawn_start >= YAWN_SEC:
            reasons.append("YAWN")

        # 끄덕임: 최근 2초 중 가장 든 자세보다 NOD_DEG 이상 숙임 (숙이면 pitch가 커진다)
        m = np.array(result.facial_transformation_matrixes[0])
        pitch = math.degrees(math.atan2(m[2, 1], m[2, 2]))
        self.pitches.append((t, pitch))
        while t - self.pitches[0][0] > NOD_SEC:
            self.pitches.popleft()
        if pitch - min(p for _, p in self.pitches) >= NOD_DEG:
            reasons.append("NOD")

        return box, reasons
