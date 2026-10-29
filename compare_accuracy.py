# ===============================================================
# TEST ACCURACY COMPARISON
# Custom CNN vs ResNet18 vs MobileNetV2 vs Ensemble
# ===============================================================

import os
import torch
from PIL import ImageFile

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ===============================================================
# PATHS
# ===============================================================

TEST_DIR = r"C:\Users\sansk\Downloads\archive (2)\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification\test"

CNN_CHECKPOINT = "my_cnn_model.pth"
MOBILENET_CHECKPOINT = "mobilenet_model.pth"
RESNET_CHECKPOINT = "resnet_model.pth"

BATCH_SIZE = 8

# ===============================================================
# DEVICE
# ===============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================================================
# IMAGE TRANSFORM
# ===============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# ===============================================================
# LOAD TEST DATASET
# ===============================================================

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"Test dataset folder not found:\n{TEST_DIR}\n"
        "Please check the TEST_DIR path."
    )

test_data = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

if len(test_data) == 0:
    raise RuntimeError("No test images found in the dataset folder.")

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Class-to-index mapping:", test_data.class_to_idx)
print("Total test images:", len(test_data))

# ===============================================================
# CUSTOM CNN MODEL
# Same architecture as model1.py
# ===============================================================

class MyCNN(nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# ===============================================================
# LOAD MODEL CHECKPOINTS
# ===============================================================

def load_checkpoint(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Place the checkpoint in the same folder as this script "
            "or update its path."
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=True
    )

# Custom CNN
cnn = MyCNN().to(device)
cnn.load_state_dict(load_checkpoint(CNN_CHECKPOINT))
cnn.eval()

print("Custom CNN loaded successfully.")

# ResNet18
resnet = models.resnet18(weights=None)

resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(resnet.fc.in_features, 2)
)

resnet.load_state_dict(load_checkpoint(RESNET_CHECKPOINT))
resnet = resnet.to(device)
resnet.eval()

print("ResNet18 loaded successfully.")

# MobileNetV2
mobile = models.mobilenet_v2(weights=None)

mobile.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(mobile.last_channel, 2)
)

mobile.load_state_dict(load_checkpoint(MOBILENET_CHECKPOINT))
mobile = mobile.to(device)
mobile.eval()

print("MobileNetV2 loaded successfully.")

# ===============================================================
# CALCULATE ACCURACY
# ===============================================================

def evaluate_models():

    correct_cnn = 0
    correct_resnet = 0
    correct_mobile = 0
    correct_ensemble = 0

    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Individual model predictions
            out_cnn = cnn(images)
            out_resnet = resnet(images)
            out_mobile = mobile(images)

            pred_cnn = out_cnn.argmax(dim=1)
            pred_resnet = out_resnet.argmax(dim=1)
            pred_mobile = out_mobile.argmax(dim=1)

            # Convert outputs to probabilities
            probs_cnn = torch.softmax(out_cnn, dim=1)
            probs_resnet = torch.softmax(out_resnet, dim=1)
            probs_mobile = torch.softmax(out_mobile, dim=1)

            # Ensemble: average the three probability distributions
            ensemble_probs = (
                probs_cnn + probs_resnet + probs_mobile
            ) / 3.0

            pred_ensemble = ensemble_probs.argmax(dim=1)

            # Count correct predictions
            total += labels.size(0)

            correct_cnn += (pred_cnn == labels).sum().item()
            correct_resnet += (pred_resnet == labels).sum().item()
            correct_mobile += (pred_mobile == labels).sum().item()
            correct_ensemble += (
                pred_ensemble == labels
            ).sum().item()

    if total == 0:
        raise RuntimeError("No images were evaluated.")

    # ===========================================================
    # FINAL RESULTS
    # ===========================================================

    print("\n======================================")
    print("       TEST ACCURACY COMPARISON")
    print("======================================")

    print(f"Custom CNN     : {100 * correct_cnn / total:.2f}%")
    print(f"ResNet18       : {100 * correct_resnet / total:.2f}%")
    print(f"MobileNetV2    : {100 * correct_mobile / total:.2f}%")
    print(f"Ensemble Model : {100 * correct_ensemble / total:.2f}%")

    print("--------------------------------------")
    print("Total test images:", total)
    print("======================================")


# ===============================================================
# RUN
# ===============================================================

if __name__ == "__main__":
    evaluate_models()