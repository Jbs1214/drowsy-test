# 졸음운전 감지 테스트

여러 모델로 졸음운전 감지를 테스트하는 저장소다. 모델마다 폴더 하나를 쓰고, 각 폴더는 독립적으로 설치하고 실행한다.

## 폴더

| 폴더 | 모델 | 설명 |
|---|---|---|
| mobilenetv2 | MobileNetV2 + MediaPipe EAR/MAR | 영상 파일 추론(`run_video.py`)과 테스트셋 평가(`eval_test.py`)를 더한 OmerKhan33 파이프라인 |

## 실행

각 폴더의 `README.md`를 따른다. 공통 흐름은 아래와 같다.

```bash
git pull
cd <모델 폴더>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 새 모델 폴더 추가 규칙

다른 모델을 테스트할 때는 기존 폴더를 고치지 말고 새 폴더를 만든다.

1. `feat/<모델명>` 브랜치를 만들고, 루트에 `<모델명>/` 폴더를 만든다.
2. 폴더 안에 아래 파일을 둔다.
   - `requirements.txt`: 동작을 확인한 버전을 `==`로 고정한다. `pip freeze`로 확인할 수 있다.
   - `README.md`: 설치 방법, 실행 방법, 판정 기준, 한계를 적는다.
3. 가상환경(`.venv`)은 폴더마다 따로 만든다. 모델마다 라이브러리 버전이 달라도 서로 충돌하지 않는다.
4. 가중치와 데이터셋은 GitHub에 올리지 않는다. 자동 다운로드 코드를 넣거나 Google Drive로 공유하고, 받는 방법을 README에 적는다.
5. 위 폴더 표에 한 줄을 추가한다.

## 올리지 않는 파일

`.gitignore`에서 모든 폴더에 공통으로 제외한다.

- `.venv/` (가상환경)
- `models/`, `*.pt`, `*.pth`, `*.onnx` (가중치)
- `data/` (데이터셋)
- `.env`, `*.key` (비밀 키)
