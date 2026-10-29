# =========================================================
# FINAL CLEAN GUI
# ENSEMBLE + GRADCAM
# NO SCROLL ISSUES
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, models

from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk, ImageFile

import cv2
import numpy as np

# =========================================================
# FIX CORRUPTED IMAGES
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
# TRANSFORM
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
# CUSTOM CNN
# =========================================================

class MyCNN(nn.Module):

    def __init__(self):

        super(MyCNN,self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            32,
            3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32,
            64,
            3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            64,
            128,
            3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2,2)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(
            128*16*16,
            256
        )

        self.fc2 = nn.Linear(
            256,
            2
        )

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

        # =================================================
        # SAVE FEATURES FOR GRADCAM
        # =================================================

        self.feature_maps = x

        self.feature_maps.retain_grad()

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
    torch.load(
        "my_cnn_model.pth",
        map_location=device
    )
)

cnn_model.eval()

print("✅ Custom CNN Loaded")

# =========================================================
# LOAD MOBILENET
# =========================================================

mobilenet_model = models.mobilenet_v2(weights=None)

mobilenet_model.classifier[1] = nn.Linear(
    mobilenet_model.last_channel,
    2
)

mobilenet_model.load_state_dict(
    torch.load(
        "mobilenet_model.pth",
        map_location=device
    )
)

mobilenet_model = mobilenet_model.to(device)

mobilenet_model.eval()

print("✅ MobileNet Loaded")

# =========================================================
# LOAD RESNET
# =========================================================

resnet_model = models.resnet18(weights=None)

resnet_model.fc = nn.Sequential(

    nn.Dropout(0.5),

    nn.Linear(
        resnet_model.fc.in_features,
        2
    )
)

resnet_model.load_state_dict(
    torch.load(
        "resnet_model.pth",
        map_location=device
    )
)

resnet_model = resnet_model.to(device)

resnet_model.eval()

print("✅ ResNet Loaded")

# =========================================================
# GRADCAM FUNCTION
# =========================================================

def generate_gradcam(model, image_tensor, image_path):

    output = model(image_tensor)

    pred_class = output.argmax(dim=1)

    score = output[:, pred_class]

    model.zero_grad()

    score.backward()

    gradients = model.feature_maps.grad

    activations = model.feature_maps

    pooled_gradients = torch.mean(
        gradients,
        dim=[0,2,3]
    )

    for i in range(128):

        activations[:,i,:,:] *= pooled_gradients[i]

    heatmap = torch.mean(
        activations,
        dim=1
    ).squeeze()

    heatmap = np.maximum(
        heatmap.detach().cpu().numpy(),
        0
    )

    if np.max(heatmap) != 0:

        heatmap /= np.max(heatmap)

    img = cv2.imread(image_path)

    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed = cv2.addWeighted(
        img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return superimposed

# =========================================================
# GUI
# =========================================================

root = Tk()

root.title(
    "AI Bone Fracture Detection"
)

root.geometry("1000x900")

root.configure(bg="#f4f6f9")

# =========================================================
# MAIN FRAME
# =========================================================

main_frame = Frame(
    root,
    bg="#f4f6f9"
)

main_frame.pack(
    fill="both",
    expand=True
)

# =========================================================
# TITLE
# =========================================================

heading = Label(

    main_frame,

    text="AI Based Bone Fracture Detection System",

    font=("Arial",24,"bold"),

    fg="#1e3a8a",

    bg="#f4f6f9"
)

heading.pack(pady=20)

subheading = Label(

    main_frame,

    text="Ensemble Learning with Grad-CAM",

    font=("Arial",14),

    fg="#374151",

    bg="#f4f6f9"
)

subheading.pack(pady=5)

# =========================================================
# IMAGE LABEL
# =========================================================

image_label = Label(

    main_frame,

    bg="#dbeafe",

    bd=3,

    relief=RIDGE
)

image_label.pack(pady=10)

# =========================================================
# RESULT LABEL
# =========================================================

result_label = Label(

    main_frame,

    text="Upload X-ray Image",

    font=("Arial",16,"bold"),

    bg="white",

    fg="#111827",

    padx=20,

    pady=20,

    width=35,

    height=4,

    relief=RIDGE
)

result_label.pack(pady=15)

# =========================================================
# GRADCAM TITLE
# =========================================================

grad_title = Label(

    main_frame,

    text="Grad-CAM Visualization",

    font=("Arial",18,"bold"),

    fg="#991b1b",

    bg="#f4f6f9"
)

grad_title.pack(pady=10)

# =========================================================
# GRADCAM LABEL
# =========================================================

gradcam_label = Label(

    main_frame,

    bg="#fee2e2",

    bd=3,

    relief=RIDGE
)

gradcam_label.pack(pady=10)

# =========================================================
# IMAGE PATH
# =========================================================

img_path = None

# =========================================================
# UPLOAD IMAGE
# =========================================================

def upload_image():

    global img_path

    img_path = filedialog.askopenfilename(

        filetypes=[
            ("Image Files","*.jpg *.png *.jpeg")
        ]
    )

    if img_path:

        img = Image.open(img_path)

        img = img.resize((220,220))

        photo = ImageTk.PhotoImage(img)

        image_label.configure(
            image=photo
        )

        image_label.image = photo

# =========================================================
# PREDICT IMAGE
# =========================================================

def predict_image():

    global img_path

    if img_path is None:

        result_label.config(
            text="Please Upload Image First"
        )

        return

    img = Image.open(
        img_path
    ).convert("RGB")

    input_tensor = transform(img)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    input_tensor.requires_grad = True

    # =====================================================
    # CNN
    # =====================================================

    cnn_output = cnn_model(input_tensor)

    cnn_probs = torch.softmax(
        cnn_output,
        dim=1
    )

    # =====================================================
    # MOBILENET
    # =====================================================

    mobile_output = mobilenet_model(input_tensor)

    mobile_probs = torch.softmax(
        mobile_output,
        dim=1
    )

    # =====================================================
    # RESNET
    # =====================================================

    resnet_output = resnet_model(input_tensor)

    resnet_probs = torch.softmax(
        resnet_output,
        dim=1
    )

    # =====================================================
    # ENSEMBLE
    # =====================================================

    ensemble_probs = (

        cnn_probs +

        mobile_probs +

        resnet_probs

    ) / 3

    conf,pred = torch.max(
        ensemble_probs,
        1
    )

    result = class_names[
        pred.item()
    ]

    confidence = conf.item()*100

    result_label.config(

        text=f"Prediction : {result}\n\nConfidence : {confidence:.2f}%"
    )

    # =====================================================
    # GRADCAM
    # =====================================================

    gradcam_image = generate_gradcam(

        cnn_model,

        input_tensor,

        img_path
    )

    gradcam_image = cv2.cvtColor(
        gradcam_image,
        cv2.COLOR_BGR2RGB
    )

    grad_img = Image.fromarray(
        gradcam_image
    )

    grad_img = grad_img.resize(
        (220,220)
    )

    grad_img = ImageTk.PhotoImage(
        grad_img
    )

    gradcam_label.configure(
        image=grad_img
    )

    gradcam_label.image = grad_img

# =========================================================
# BUTTONS
# =========================================================

upload_btn = Button(

    main_frame,

    text="Upload X-Ray",

    font=("Arial",14,"bold"),

    bg="#60a5fa",

    fg="white",

    padx=20,

    pady=10,

    command=upload_image
)

upload_btn.pack(pady=10)

predict_btn = Button(

    main_frame,

    text="Predict",

    font=("Arial",14,"bold"),

    bg="#22c55e",

    fg="white",

    padx=20,

    pady=10,

    command=predict_image
)

predict_btn.pack(pady=10)

# =========================================================
# RUN GUI
# =========================================================

root.mainloop()