import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision import transforms
import matplotlib.pyplot as plt

# === Load Model ===
model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
model.classifier[4] = nn.LazyConv2d(2, 1)
model.aux_classifier[4] = nn.LazyConv2d(2, 1)
model.load_state_dict(torch.load("document_segmenter.pth", weights_only=True))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Image Path ===
image_path = "DewarpBook.jpg"
orig = cv2.imread(image_path)
resized = cv2.resize(orig, (384, 384))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# === Preprocess ===
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109))
])
tensor_img = preprocess(rgb).unsqueeze(0).to(device)

# === Inference ===
with torch.no_grad():
    out = model(tensor_img)["out"]
    pred = torch.argmax(out.squeeze(), dim=0).cpu().numpy()

# === Post-process mask ===
mask = (pred * 255).astype(np.uint8)
mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

# Clean the mask
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
mask_clean = cv2.dilate(mask_clean, np.ones((5, 5), np.uint8), iterations=1)

# === Contour Detection and Warping ===
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Find largest contour
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No contours found in mask.")
    exit()

contour = max(contours, key=cv2.contourArea)
peri = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

if len(approx) == 4:
    print("Found 4-point contour")
    pts = approx.reshape(4, 2)
else:
    print("⚠Couldn't find 4-point polygon, using convex hull + minAreaRect")
    hull = cv2.convexHull(contour)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    pts = box.astype(np.intp)


# Warp the image
scanned = four_point_transform(orig, pts)
cv2.imwrite("scanned_output.jpg", scanned)
print("Scanned document saved as 'scanned_output.jpg'")
