# mobilevit — MobileViT v2 졸음 감지

PC 웹캠으로 얼굴을 실시간으로 보고, 졸음이 감지되면 화면 경고와 경고음을 낸다.
모델은 Hugging Face [mosesb/drowsiness-detection-mobileViT-v2](https://huggingface.co/mosesb/drowsiness-detection-mobileViT-v2) (timm `mobilevitv2_200`, Drowsy / Non Drowsy 2클래스)를 그대로 사용한다.

## 설치

이 폴더 안에서 실행한다.

```bash
cd mobilevit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Python 3.12, Windows 기준으로 확인했다.
- 라이브러리 버전은 `requirements.txt`에 `==`로 고정되어 있다. 동작을 확인한 버전이므로 임의로 올리지 않는다.
  - 특히 `opencv-python`은 5 미만이어야 한다. OpenCV 5에는 얼굴 검출에 쓰는 `cv2.CascadeClassifier`가 없다.
- `torch`는 CPU 빌드로 설치된다. NVIDIA GPU를 쓰려면 [PyTorch 설치 페이지](https://pytorch.org/get-started/locally/)의 CUDA 설치 명령으로 같은 버전을 설치한다. 현재 코드는 CPU로 추론한다.
- 가중치(`models/best_model.pt`, 약 70MB)는 처음 실행할 때 Hugging Face에서 자동으로 다운로드된다. GitHub에는 올리지 않는다.

## 실행

```bash
python app.py                 # 파일 선택 창에서 영상 선택, 취소하면 웹캠 실시간 감지. q로 종료
python predict.py face.jpg    # 이미지 1장 판정
python test_videos.py ../sleep test_results.csv   # 폴더 안 mp4 전체를 화면 없이 판정해 CSV로 저장
```

## 판정 기준

- 매 프레임에서 가장 큰 얼굴을 잘라 Drowsy 확률을 구한다.
- 최근 2초간 확률 평균이 0.7을 넘으면 졸음으로 판정한다.
  - 화면에 빨간 테두리와 `DROWSY! WAKE UP` 문구를 띄운다.
  - 경고음은 1초에 한 번 울린다.
- 얼굴이 검출되지 않으면 기록을 비우므로 경고가 나지 않는다.
- 기준값은 `app.py` 상단의 `WINDOW_SEC`, `THRESHOLD`에서 바꾼다.

## 영상 테스트 결과 (2026-10-29)

### 조건

- 데이터: `sleep/` 폴더의 차량 내 운전자 영상 160개(운전자 8명 × 20개, 1280×720, 30fps, 3.7~17.6초, 총 23분)와 라벨(`sleep/TL1/**/label/*.json`의 `scene_info.category_name`). 데이터는 GitHub에 올리지 않는다(`.gitignore`의 `sleep/`).
- `졸음운전` 80개를 졸음, 나머지 80개(음주운전 49, 물건찾기 30, 휴대폰조작 1)를 졸음 아님으로 보고 평가했다. 라벨과 데이터 오류는 [루트 README](../README.md#모델별-비교)에 정리했다.
- 판정 기준은 `app.py`와 같다(Haar 얼굴 검출 → 2초 평균 Drowsy 확률 > 0.7). 30fps 영상을 3프레임마다 1장씩(10fps) 판정했고, 영상 안에서 한 번이라도 경고가 나면 "경고"로 셌다.
- 영상별 결과: [`test_results.csv`](test_results.csv)
  - `face_ratio`: 얼굴이 검출된 프레임 비율
  - `mean_prob`: 얼굴이 검출된 프레임의 평균 Drowsy 확률
  - `drowsy_ratio`: 졸음 경고 상태였던 프레임 비율
  - `alerted`, `first_alert_sec`: 경고 발생 여부와 첫 경고 시각(초)

### 요약

| 항목 | 결과 |
|---|---|
| 졸음 감지 | 33 / 80 (41%) |
| 오경보 | 31 / 80 (39%) — 음주운전 23/49, 물건찾기 8/30, 휴대폰조작 0/1 |
| 정확도 | 51% |

| 라벨 | 얼굴 검출 비율 | 평균 Drowsy 확률 |
|---|---|---|
| 졸음운전 | 0.70 | 0.73 |
| 음주운전 | 0.60 | 0.72 |
| 물건찾기 | 0.52 | 0.65 |

### 해석

- **모델이 졸음과 다른 행동을 구분하지 못한다.** 라벨별 평균 Drowsy 확률이 거의 같고, 정확도 51%는 찍는 것과 비슷하다.
- 경고가 덜 나는 주원인은 얼굴 검출 실패다. Haar 얼굴 검출이 1초에 한두 번씩 실패하고, 실패할 때마다 판정 기록을 비우므로 2초 구간이 채워지지 않는다.
  - 예: `SGA2100313S0028.mp4`는 평균 확률 0.999인데 얼굴이 1초 안팎마다 한 번씩 끊겨 경고가 나지 않았다.
- 학습 데이터(Kaggle 운전자 얼굴 이미지)와 이 데이터(차 안 측면 각도, 안경, 머리의 뇌파 센서)의 촬영 환경이 달라서 모델 카드의 정확도(98%)가 재현되지 않는 것으로 보인다.
- 얼굴 랜드마크 규칙(눈 감음·하품·고개 끄덕임)과의 비교는 [루트 README](../README.md#모델별-비교)를 본다.

## 한계

- 모델 카드에 따르면 연구용 proof-of-concept이다. 실제 안전 장치로 단독 사용하면 안 된다.
- 야간·어두운 조명, 선글라스·모자·손 가림, 정면이 아닌 카메라 각도에서는 정확도가 떨어질 수 있다.
- CPU 추론은 약 140ms/프레임이라서 FPS가 5~6 정도로 낮다.
- 영상 파일은 모든 프레임을 추론하므로 실제 속도보다 느리게 재생된다. 2초 판정 구간은 영상 시간 기준이다.
