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
python app.py                 # 웹캠 실시간 감지, q로 종료
python predict.py face.jpg    # 이미지 1장 판정
```

## 판정 기준

- 매 프레임에서 가장 큰 얼굴을 잘라 Drowsy 확률을 구한다.
- 최근 2초간 확률 평균이 0.7을 넘으면 졸음으로 판정한다.
  - 화면에 빨간 테두리와 `DROWSY! WAKE UP` 문구를 띄운다.
  - 경고음은 1초에 한 번 울린다.
- 얼굴이 검출되지 않으면 기록을 비우므로 경고가 나지 않는다.
- 기준값은 `app.py` 상단의 `WINDOW_SEC`, `THRESHOLD`에서 바꾼다.

## 한계

- 모델 카드에 따르면 연구용 proof-of-concept이다. 실제 안전 장치로 단독 사용하면 안 된다.
- 야간·어두운 조명, 선글라스·모자·손 가림, 정면이 아닌 카메라 각도에서는 정확도가 떨어질 수 있다.
- CPU 추론은 약 140ms/프레임이라서 FPS가 5~6 정도로 낮다.
