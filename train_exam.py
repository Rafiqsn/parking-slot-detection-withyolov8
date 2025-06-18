# 2. Import
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root_dir)
        self.transform = transform
        self.class_to_indices = self._prepare_class_index()

    def _prepare_class_index(self):
        class_to_indices = {}
        for idx, (_, label) in enumerate(self.data.samples):
            class_to_indices.setdefault(label, []).append(idx)
        return class_to_indices

    def __getitem__(self, index):
        anchor_img, anchor_label = self.data[index]

        # Positive sample
        pos_indices = self.class_to_indices[anchor_label].copy()
        pos_indices.remove(index)
        pos_index = random.choice(pos_indices) if pos_indices else index
        positive_img, _ = self.data[pos_index]

        # Negative sample
        neg_label = random.choice(
            [l for l in self.class_to_indices.keys() if l != anchor_label]
        )
        neg_index = random.choice(self.class_to_indices[neg_label])
        negative_img, _ = self.data[neg_index]

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.data)


class EmbedderNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()
        self.backbone = base
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # important for cosine
        return x


# Hyperparameter & transform
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# Load dataset
dataset = TripletDataset(
    "/content/drive/MyDrive/ParkingDetection/cropped_dataset/train", transform=transform
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model & loss
model = EmbedderNet(emb_dim=128).to(device)
criterion = nn.TripletMarginLoss(margin=0.5, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for anchor, positive, negative in tqdm(dataloader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = criterion(anchor_emb, positive_emb, negative_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), "embedder_triplet.pt")
