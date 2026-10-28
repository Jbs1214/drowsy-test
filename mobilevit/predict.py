import os
import sys

import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

REPO_ID = "mosesb/drowsiness-detection-mobileViT-v2"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "best_model.pt")
CLASS_NAMES = ["Drowsy", "Non Drowsy"]  # index 0 = Drowsy

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        hf_hub_download(repo_id=REPO_ID, filename="best_model.pt", local_dir=MODELS_DIR)
    model = timm.create_model("mobilevitv2_200", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


def predict_drowsy(model, pil_image):
    """Drowsy 클래스 확률(0~1)을 반환한다."""
    x = preprocess(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    return float(probs[0])


if __name__ == "__main__":
    model = load_model()
    p = predict_drowsy(model, Image.open(sys.argv[1]))
    label = CLASS_NAMES[0] if p > 0.5 else CLASS_NAMES[1]
    print(f"{label} (Drowsy 확률: {p:.3f})")
