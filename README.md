# document_segmenter
A PyTorch-based semantic segmentation model for document detection, trained on a fully synthetic dataset. This model learns to detect document regions from natural images containing documents over complex backgrounds.

📦 Features:
- DeepLabV3 + ResNet50 or MobileNetV3 backbone

- Synthetic dataset generation using real documents + background textures

- Robust augmentations (blur, shadow, elastic distortion, etc.)

- Custom loss with Dice + Binary Cross Entropy + Total Variation for smooth, accurate masks

- Output masks are clean and hole-free, suitable for downstream scanning or OCR tasks
