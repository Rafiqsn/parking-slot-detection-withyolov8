import torch
from torchvision import models

# Buat objek model sesuai arsitektur yang digunakan saat training
model = models.mobilenet_v2(
    num_classes=models.mobilenet_v2.NUM_CLASSES  # Ganti NUM_CLASSES sesuai model Anda
)

# Load state_dict ke model
state_dict = torch.load(
    "models/embedding_model_mobilenetv2.pth", map_location=torch.device("cpu")
)
model.load_state_dict(state_dict)
model.eval()

# Dummy input sesuai input model Anda
dummy_input = torch.randn(1, 3, 128, 64)  # ganti sesuai kebutuhan

# Ekspor ke ONNX
torch.onnx.export(model, dummy_input, "models/embedding_model_mobilenetv2.onnx")
