"""
Shared preprocessing for the real-vs-AI detector.

Imported by BOTH trainPyTorch_v2.py and detection.py so the model sees
identical inputs at training time and inference time. Do not duplicate
these functions anywhere else.
"""
import cv2
import numpy as np
import torch
from torchvision import transforms

IMG_SIZE = 224
_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def build_val_transform(img_size=IMG_SIZE):
    """Deterministic RGB transform (no augmentation) used for validation and inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        _normalize,
    ])


def fft_features(img_rgb, img_size=IMG_SIZE):
    """Log-magnitude FFT spectrum, z-score normalized, repeated to 3 channels."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    freq = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(freq) + 1.0)
    mag = cv2.resize(mag, (img_size, img_size))
    mag = (mag - mag.mean()) / (mag.std() + 1e-8)
    mag = np.repeat(mag[np.newaxis, ...], 3, axis=0)
    return torch.tensor(mag, dtype=torch.float32)


def noise_residual(img_rgb, img_size=IMG_SIZE):
    """Signed noise residual (image minus 5x5 Gaussian blur), scaled to about [-1, 1]."""
    blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    residual = img_rgb.astype(np.float32) - blur.astype(np.float32)
    residual = cv2.resize(residual, (img_size, img_size)) / 127.5
    return torch.tensor(residual).permute(2, 0, 1).float()
