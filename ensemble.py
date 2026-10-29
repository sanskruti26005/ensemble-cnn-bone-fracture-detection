# =========================================================
# FINAL ENSEMBLE MODEL - FIXED 300% BUG
# CUSTOM CNN + MOBILENETV2 + RESNET18
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageFile

# =========================================================
# FIX CORRUPTED IMAGE ISSUE
# =========================================================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================
# DEVICE
# =========================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using Device:", device)

# =========================================================
# CLASS NAMES
# =========================================================
class_names = [
    "Fractured",
    "Non-Fractured"
]

# =========================================================
# IMAGE TRANSFORM
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
# CUSTOM CNN MODEL
# =========================================================
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*16*16,256)
        self.fc2 = nn.Linear(256,2)

    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =========================================================
# LOAD CUSTOM CNN
# =========================================================
cnn_model = MyCNN().to(device)
cnn_model.load_state_dict(
    torch.load("my_cnn_model.pth", map_location=device)
)
cnn_model.eval()
print("✅ Custom CNN Loaded")

# =========================================================
# LOAD MOBILENETV2
# =========================================================
mobilenet_model = models.mobilenet_v2(weights=None)
# FIX: Make sure classifier matches training
mobilenet_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(mobilenet_model.last_channel, 2)
)
mobilenet_model.load_state_dict(
    torch.load("mobilenet_model.pth", map_location=device)
)
mobilenet_model = mobilenet_model.to(device)
mobilenet_model.eval()
print("✅ MobileNetV2 Loaded")

# =========================================================
# LOAD RESNET18
# =========================================================
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(resnet_model.fc.in_features, 2)
)
resnet_model.load_state_dict(
    torch.load("resnet_model.pth", map_location=device)
)
resnet_model = resnet_model.to(device)
resnet_model.eval()
print("✅ ResNet18 Loaded")

# =========================================================
# IMAGE PATH - CHANGE THIS
# =========================================================
img_path = r"C:\Users\sansk\OneDrive\Pictures\Documents\examples\non fracture\pexels-gayan-kavinda-2157879009-35053995.jpg"

# =========================================================
# LOAD IMAGE
# =========================================================
img = Image.open(img_path).convert("RGB")
img = transform(img)
img = img.unsqueeze(0).to(device)

# =========================================================
# CUSTOM CNN PREDICTION
# =========================================================
with torch.no_grad():
    cnn_output = cnn_model(img)
    cnn_probs = torch.softmax(cnn_output, dim=1)
    cnn_conf,cnn_pred = torch.max(cnn_probs, 1)

# =========================================================
# MOBILENETV2 PREDICTION
# =========================================================
with torch.no_grad():
    mobile_output = mobilenet_model(img)
    mobile_probs = torch.softmax(mobile_output, dim=1)
    mobile_conf,mobile_pred = torch.max(mobile_probs, 1)

# =========================================================
# RESNET18 PREDICTION
# =========================================================
with torch.no_grad():
    resnet_output = resnet_model(img)
    resnet_probs = torch.softmax(resnet_output, dim=1)
    resnet_conf,resnet_pred = torch.max(resnet_probs, 1)

# =========================================================
# ENSEMBLE AVERAGING - FIXED ✅
# BEFORE: cnn_probs + mobile_probs + resnet_probs = 300%
# AFTER:  / 3.0 = 100%
# =========================================================
ensemble_probs = (
    cnn_probs +
    mobile_probs +
    resnet_probs
) / 3.0

ensemble_conf,ensemble_pred = torch.max(
    ensemble_probs,
    1
)

# =========================================================
# PRINT RESULTS
# =========================================================
print("\n==============================")
print("INDIVIDUAL MODEL RESULTS")
print("==============================")
print(f"\nCustom CNN : {class_names[cnn_pred.item()]} ({cnn_conf.item()*100:.2f}%)")
print(f"\nMobileNetV2 : {class_names[mobile_pred.item()]} ({mobile_conf.item()*100:.2f}%)")
print(f"\nResNet18 : {class_names[resnet_pred.item()]} ({resnet_conf.item()*100:.2f}%)")

print("\n==============================")
print("FINAL ENSEMBLE RESULT")
print("==============================")
print(f"\nPrediction : {class_names[ensemble_pred.item()]}")
print(f"Confidence : {ensemble_conf.item()*100:.2f}%") # Now will be <=100%
