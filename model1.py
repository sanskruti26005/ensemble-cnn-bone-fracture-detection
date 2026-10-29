# =========================================================
# FINAL CUSTOM CNN
# BETTER GENERALIZATION VERSION
# TARGET:
# Train Accuracy  : 93%
# Test Accuracy   : 91%
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

# =========================================================
# FIX CORRUPTED IMAGES
# =========================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_loader(path):

    try:
        return Image.open(path).convert("RGB")

    except:

        print("Skipping Corrupted Image:", path)

        return Image.new("RGB",(128,128))

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# =========================================================
# DATA AUGMENTATION
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

train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True
)

# =========================================================
# FINAL CUSTOM CNN
# =========================================================

class MyCNN(nn.Module):

    def __init__(self):

        super(MyCNN,self).__init__()

        # -------------------------------------------------
        # CONV BLOCK 1
        # -------------------------------------------------

        self.conv1 = nn.Conv2d(
            3,
            32,
            3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        # -------------------------------------------------
        # CONV BLOCK 2
        # -------------------------------------------------

        self.conv2 = nn.Conv2d(
            32,
            64,
            3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        # -------------------------------------------------
        # CONV BLOCK 3
        # -------------------------------------------------

        self.conv3 = nn.Conv2d(
            64,
            128,
            3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        # -------------------------------------------------

        self.pool = nn.MaxPool2d(2,2)

        self.dropout = nn.Dropout(0.5)

        # -------------------------------------------------
        # FULLY CONNECTED
        # -------------------------------------------------

        self.fc1 = nn.Linear(
            128*16*16,
            256
        )

        self.fc2 = nn.Linear(
            256,
            2
        )

    # =====================================================
    # FORWARD FUNCTION
    # =====================================================

    def forward(self,x):

        # -------------------------------------------------
        # BLOCK 1
        # -------------------------------------------------

        x = self.pool(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        # -------------------------------------------------
        # BLOCK 2
        # -------------------------------------------------

        x = self.pool(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        # -------------------------------------------------
        # BLOCK 3
        # -------------------------------------------------

        x = self.pool(
            F.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        # -------------------------------------------------

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x

# =========================================================
# MODEL
# =========================================================

model = MyCNN().to(device)

# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.00005
)

# =========================================================
# LEARNING RATE SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# =========================================================
# TRAINING
# =========================================================

epochs = 10

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

    # -----------------------------------------------------
    # EPOCH RESULTS
    # -----------------------------------------------------

    acc = 100 * correct / total

    avg_loss = running_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Completed")

    print(f"Accuracy : {acc:.2f}%")

    print(f"Average Loss : {avg_loss:.4f}\n")

    # -----------------------------------------------------
    # UPDATE LEARNING RATE
    # -----------------------------------------------------

    scheduler.step()

# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "my_cnn_model.pth"
)

print("✅ Final Custom CNN Model Saved")