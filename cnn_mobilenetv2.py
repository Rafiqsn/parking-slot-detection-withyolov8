import torch
import torchvision.models as models
import torch.nn as nn

from torchvision import transforms
from PIL import Image
import cv2  # Jangan lupa ini kalau pakai OpenCV
import numpy as np


# 1. Preprocessing transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # standard imagenet
        ),
    ]
)


# 2. Load feature extractor (MobileNetV2 + GAP)
def create_mobilenetv2_feature_extractor():
    model = models.mobilenet_v2(pretrained=True)
    feature_extractor = nn.Sequential(
        model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
    )
    feature_extractor.eval()
    return feature_extractor


# 3. Ekstrak fitur dari bounding box
def extract_feature_from_bbox(image, bbox, model):
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]  # asumsi image = numpy (OpenCV)
    img_pil = Image.fromarray(
        cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    )  # convert ke RGB
    input_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        features = model(input_tensor)

    return features.squeeze().numpy()


# 4. Tes pakai gambar contoh
if __name__ == "__main__":
    model = create_mobilenetv2_feature_extractor()

    # Load gambar/frame dan bbox contoh
    frame = cv2.imread("./1.png")
    bbox = [227, 145, 280, 196]  # kamu ganti dengan hasil deteksi YOLO

    feature = extract_feature_from_bbox(frame, bbox, model)
    print("Shape feature vector:", feature.shape)  # Harusnya (1280,)
