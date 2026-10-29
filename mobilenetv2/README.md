# mobilenetv2

MobileNetV2 분류기와 MediaPipe 얼굴 랜드마크(EAR/MAR)를 합쳐 졸음을 판정한다. OmerKhan33의 [driver-drowsiness-detection-system](https://github.com/OmerKhan33/driver-drowsiness-detection-system)을 가져와 영상 파일 추론과 테스트셋 평가 스크립트를 더했다. 원본 README는 `README.upstream.md`에 그대로 두었다.

## 설치

검증한 환경은 conda `drowsy`(Python 3.11)이고, `requirements.txt`는 이 환경에서 `pip freeze`로 만들었다.

```bash
cd mobilenetv2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 데이터와 가중치

둘 다 GitHub에 올리지 않는다. 각자 받는다.

데이터는 Kaggle [Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)이다.

1. 받은 파일을 `dataset/<train|val|test>/<클래스>/` 구조로 푼다. 원본 README의 `data/raw`가 아니라 폴더 바로 아래 `dataset/`이다. 폴더 두 단계까지만 읽는다.
2. 클래스 매핑은 아래와 같다.
   - alert: `Open_Eyes`, `Open`, `no_yawn`
   - drowsy: `Closed_Eyes`, `Closed`, `Yawn`, `yawn`
3. 나누고 검증한다.

```bash
python data/scripts/prepare_dataset.py
python data/scripts/validate_dataset.py
```

가중치는 올리지 않으니 직접 학습한다. 폴더 루트에서 `-m`으로 실행해야 한다. `python src/classification/train.py`로 실행하면 `No module named 'src'`가 난다. `--models`를 빼면 6개 모델을 전부 학습해서 오래 걸린다.

```bash
python -m src.classification.train --models MobileNetV2 --epochs 8
```

결과는 `models/weights/MobileNetV2_best.pt`에 저장된다.

MediaPipe 얼굴 랜드마크 모델(`models/face_landmarker.task`, 약 3.6MB)도 받아야 한다. 없으면 `run_video.py`와 Streamlit 앱이 실행되지 않는다.

```bash
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## 실행

웹캠(Streamlit):

```bash
streamlit run app/streamlit_app.py
```

영상 파일(`run_video.py`):

```bash
python run_video.py --video clip.mp4 --model MobileNetV2
python run_video.py --video clip.mp4 --consec 60 --show
```

- 결과는 `video_results/`에 CSV와 표시가 들어간 mp4로 저장되고, 졸음 구간은 터미널에 출력된다.
- `--consec`는 프레임 단위다. "2초 이상 감김"으로 보려면 `fps × 2`로 준다. 30fps면 60이다.

테스트셋 평가(`eval_test.py`):

```bash
python eval_test.py --model MobileNetV2
```

`eval_results/`에 `metrics.json`, `confusion_matrix.png`, `samples.png`, `errors.png`가 저장된다.

## 판정 기준

프레임마다 아래를 계산해 합친다.

- EAR(눈 감김): 0.23 미만이면 감김 (`--ear`)
- MAR(하품): 0.60 초과면 하품 (`--mar`)
- 연속 프레임: 기본 4프레임 이상 이어지면 졸음 경보 (`--consec`)
- EAR/MAR은 지수이동평균(0.55)으로 부드럽게 만들고, 얼굴 이미지는 CLAHE로 밝기를 보정한다.
- MobileNetV2가 얼굴 크롭에서 drowsy 확률을 내고, EAR/MAR과 가중해서 점수로 만든다.
- 얼굴이 검출되지 않은 프레임은 깨어 있는 값(EAR 0.30, MAR 0.40)으로 처리한다.

## 측정 결과

테스트 437장 (`eval_results/metrics.json`), MobileNetV2, 3에폭 학습.

| | 예측 alert | 예측 drowsy |
|---|---|---|
| 실제 alert | 217 | 2 |
| 실제 drowsy | 33 | 185 |

- 정확도 0.920
- alert 재현율 0.991, drowsy 재현율 0.849, drowsy 정밀도 0.989

### 차량 안 영상 시험 (AI Hub 샘플)

AI Hub 「운전자 및 탑승자 상태 및 이상행동 모니터링」 샘플(`New_Sample_운전자(abnormal)`, 영상 160개)로 시험했다. 학습은 하지 않고 위 모델을 그대로 돌렸다.

- 영상은 1280×720, 30fps, 4~8초다. `A001 졸음운전` 80개(피험자 5명)와 음주운전·물건찾기·휴대폰조작 영상이 섞여 있다.
- 프레임마다 행동 라벨이 있다. 졸음 영상에는 `꾸벅꾸벅졸다`, `하품` 프레임이 들어 있다.
- 영상 10개만 돌렸다. `--consec 60`(2초) 기준이다.

| 영상 | 개수 | 얼굴 검출률 | 경고 |
|---|---|---|---|
| 졸음운전(`SGA2100300`) | 5 | 91~99% | 0건 |
| 그 밖의 이상행동(`SGA2100302`) | 5 | 83~100% | 1건 |

- 졸음운전 영상 5개에서 경고가 한 번도 나오지 않았다. 이 시험에서는 졸음을 구분하지 못했다.
- 원인은 확인하지 못했다. 학습 데이터(눈이 크게 보이는 근접 얼굴)와 차량 안 영상의 차이일 수도 있고, 졸음 동작이 짧아 연속 60프레임 기준에 걸리지 않았을 수도 있다.
- 영상 10개라 수치로 쓸 수 없다. 프레임별 평가와 피험자 단위 분리는 하지 않았다.

## 한계

- 놓친 drowsy 33장은 대부분 하품 얼굴이다. 이 데이터셋은 하품을 drowsy로 라벨한다.
- 이미지 단위로 무작위 분할했다. 같은 사람이 train과 test에 섞였을 수 있어서 0.92는 높게 나온 값일 수 있다.
- 차량 안 영상(AI Hub 샘플)에서는 졸음운전 영상 5개를 모두 놓쳤다. 이 모델은 근접 얼굴 사진으로 학습해서 차량 영상에 그대로 쓰기 어렵다.
- 합성 클립(데이터셋 이미지로 만든 영상)의 얼굴 검출률은 48%였다.
- 팀 기획서(`droswy_project.md`)의 최종 판정기는 EAR·MAR·고개 각도 시계열 + 규칙 기반 + 시퀀스 분류기(1D-CNN/LSTM)다. 이 폴더의 이미지 분류기는 모델 후보 비교용이라 기획서 평가표와 같은 조건이 아니다.
- 3에폭만 학습했다.

## 올리지 않는 파일

`dataset/`, `data/`, `models/weights/*.pt`, `mlflow.db`, `mlruns/`, `eval_results/`, `video_results/`, `drowsiness-dataset.zip`. 용량이 크거나 얼굴 사진이 들어 있다.

## 출처

- 원본: [OmerKhan33/driver-drowsiness-detection-system](https://github.com/OmerKhan33/driver-drowsiness-detection-system) (MIT, 원본 README 기준)
- 이 저장소에서 더한 것: `run_video.py`, `eval_test.py`, `pip freeze`로 다시 만든 `requirements.txt`
