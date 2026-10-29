# =========================================================
# HIGHER ACCURACY + FASTER MOBILENETV2
# Optimized for CPU + Large Dataset
# =========================================================

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

# ---------------------------------------------------------
# FIX CORRUPTED IMAGES
# ---------------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_loader(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return Image.new("RGB", (128,128))

# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# =========================================================
# BETTER AUGMENTATION
# =========================================================

transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05,0.05)
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])

# =========================================================
# TRAIN DATASET
# =========================================================

train_data = datasets.ImageFolder(
r"C:\Users\sansk\Downloads\archive (2)\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification\train",
transform=transform,
loader=safe_loader
)

# ---------------------------------------------------------
# BIGGER BATCH SIZE = FASTER
# ---------------------------------------------------------

train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True
)

# =========================================================
# PRETRAINED MOBILENETV2
# =========================================================

model = models.mobilenet_v2(weights="DEFAULT")

# =========================================================
# PARTIAL FINE TUNING
# Freeze early layers
# Train later layers
# =========================================================

for param in model.features[:10].parameters():
    param.requires_grad = False

# ---------------------------------------------------------
# CUSTOM FINAL CLASSIFIER
# ---------------------------------------------------------

model.classifier = nn.Sequential(

    nn.Dropout(0.3),

    nn.Linear(
        model.last_channel,
        2
    )
)

model = model.to(device)

# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=0.0001
)

# =========================================================
# TRAINING
# =========================================================

epochs = 5

print("\nStarting Training...\n")

for epoch in range(epochs):

    model.train()

    correct = 0
    total = 0

    running_loss = 0

    for i,(images,labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _,pred = torch.max(outputs,1)

        total += labels.size(0)

        correct += (pred==labels).sum().item()

        # -------------------------------------------------
        # LIVE PROGRESS
        # -------------------------------------------------

        if i % 50 == 0:

            print(
                f"Epoch {epoch+1} | "
                f"Batch {i} | "
                f"Loss {loss.item():.4f}"
            )

    acc = 100 * correct / total

    avg_loss = running_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Completed")

    print(f"Accuracy : {acc:.2f}%")

    print(f"Average Loss : {avg_loss:.4f}\n")

# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "mobilenet_model.pth"
)

print("✅ MobileNetV2 Model Saved")