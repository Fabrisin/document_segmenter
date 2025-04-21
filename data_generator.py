import cv2
import numpy as np
import os
import random
from glob import glob
from tqdm import tqdm
from albumentations import (
    Compose, Resize, RandomBrightnessContrast, HorizontalFlip, VerticalFlip,
    Rotate, OpticalDistortion, GridDistortion, ElasticTransform,
    GaussNoise, MotionBlur, RandomShadow, RGBShift, ImageCompression
)

# === AUGMENTATION PIPELINE ===
def get_augmentations(height=384, width=384):
    return Compose([
        Resize(height, width),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=25, p=0.5),
        RandomBrightnessContrast(p=0.5),
        RGBShift(p=0.3),
        RandomShadow(p=0.3),
        GaussNoise(p=0.3),
        MotionBlur(p=0.2),
        ImageCompression(p=0.2),
        GridDistortion(p=0.3),
        ElasticTransform(p=0.3),
    ])

# === SYNTHETIC GENERATOR ===
def generate_synthetic_dataset(doc_paths, bg_paths, output_dir, num_samples=1000, image_size=(384, 384)):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)

    aug = get_augmentations(*image_size)

    for i in tqdm(range(num_samples)):
        doc_path = random.choice(doc_paths)
        bg_path = random.choice(bg_paths)

        doc_img = cv2.imread(doc_path, cv2.IMREAD_UNCHANGED)
        bg_img = cv2.imread(bg_path)

        if doc_img is None or bg_img is None:
            continue

        # Convert grayscale to RGB if needed
        if doc_img.ndim == 2:
            doc_img = cv2.cvtColor(doc_img, cv2.COLOR_GRAY2BGR)

        # Resize document and create mask
        scale = min(image_size[0] / doc_img.shape[0], image_size[1] / doc_img.shape[1])
        new_w, new_h = int(doc_img.shape[1] * scale), int(doc_img.shape[0] * scale)
        doc_img_resized = cv2.resize(doc_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = np.ones(doc_img_resized.shape[:2], dtype=np.uint8) * 255

        # Random position to place the doc
        bg_img_resized = cv2.resize(bg_img, image_size[::-1])
        canvas = bg_img_resized.copy()
        mask_canvas = np.zeros(image_size, dtype=np.uint8)

        x_offset = random.randint(0, image_size[1] - new_w)
        y_offset = random.randint(0, image_size[0] - new_h)

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = doc_img_resized
        mask_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask

        # Apply augmentations
        augmented = aug(image=canvas, mask=mask_canvas)
        final_img = augmented['image']
        final_mask = augmented['mask']

        # Save
        cv2.imwrite(f"{output_dir}/images/img_{i:05d}.jpg", final_img)
        cv2.imwrite(f"{output_dir}/masks/mask_{i:05d}.png", final_mask)

    print(f"✅ {num_samples} synthetic samples generated in {output_dir}")

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    document_folder = "C:/Users/fabri/OneDrive/Desktop/UPTP/Computer Vision/MidtermProject/Machine Learning Apporach/documents"
    background_folder = "C:/Users/fabri/OneDrive/Desktop/UPTP/Computer Vision/MidtermProject/Machine Learning Apporach/images"
    output_folder = "synthetic_dataset"

    doc_files = glob(os.path.join(document_folder, "*.jpg")) + glob(os.path.join(document_folder, "*.png"))
    bg_files = glob(os.path.join(background_folder, "*/*.jpg")) + glob(os.path.join(background_folder, "*/*.png"))

    generate_synthetic_dataset(doc_files, bg_files, output_folder, num_samples=2500)
