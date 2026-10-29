# =========================================================
# TEST FINAL CUSTOM CNN
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
# TEST TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])

# =========================================================
# TEST DATASET
# =========================================================

test_data = datasets.ImageFolder(

r"C:\Users\sansk\Downloads\archive (3)\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification\test",

transform=transform,

loader=safe_loader
)

test_loader = DataLoader(
    test_data,
    batch_size=16,
    shuffle=False
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

        x = self.pool(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.pool(
            F.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x

# =========================================================
# LOAD MODEL
# =========================================================

model = MyCNN().to(device)

model.load_state_dict(
    torch.load("my_cnn_model.pth")
)

model.eval()

print("✅ Custom CNN Loaded")

# =========================================================
# TESTING
# =========================================================

correct = 0
total = 0

print("\nStarting Testing...\n")

with torch.no_grad():

    for images,labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _,predicted = torch.max(outputs,1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

# =========================================================
# FINAL ACCURACY
# =========================================================

accuracy = 100 * correct / total

print("\n==============================")
print("FINAL TEST ACCURACY")
print("==============================")

print(f"\nTest Accuracy : {accuracy:.2f}%")