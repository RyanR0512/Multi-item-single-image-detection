import os
import io
import torch
import timm
import numpy as np
import zipfile
import requests
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFilter, ImageDraw

# ============================================================
# CONFIG
# ============================================================
YOLO_HF_URL = "https://huggingface.co/RyanR0512/Yolov11-detector/resolve/main/yolo11n.pt"
CLASSIFIER_HF_URL = "https://huggingface.co/RyanR0512/flux-classifier/resolve/main/best_model.pt"
YOLO_MODEL_PATH = "yolo11_detector.pt"
AI_MODEL_PATH = "flux_classifier.pt"
IMG_SIZE = 224
YOLO_IMAGE_SIZE = 640
AI_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TRANSFORMS
# ============================================================
rgb_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
# CV2 REPLACEMENTS (Pillow + numpy only)
# ============================================================


def pil_resize(img_np, w, h):
    """Resize a numpy array (H,W,C) or (H,W) to (h, w) using Pillow."""
    if img_np.ndim == 2:
        pil = Image.fromarray(img_np.astype(np.float32))
        pil = pil.resize((w, h), Image.BILINEAR)
        return np.array(pil)

    pil = Image.fromarray(img_np.astype(np.uint8))
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.array(pil)


def rgb_to_gray(img_rgb_np):
    """Convert RGB numpy array to grayscale using luminance weights."""
    return (
        0.299 * img_rgb_np[:, :, 0]
        + 0.587 * img_rgb_np[:, :, 1]
        + 0.114 * img_rgb_np[:, :, 2]
    ).astype(np.float32)


def gaussian_blur(img_rgb_np, radius=2):
    """Apply Gaussian blur to an RGB numpy array using Pillow."""
    pil = Image.fromarray(img_rgb_np.astype(np.uint8))
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred).astype(np.float32)


def draw_box_and_label(img_pil, x1, y1, x2, y2, label, color):
    """Draw bounding box and label onto a PIL image in-place."""
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    text_y = max(y1 - 16, 0)
    draw.text((x1, text_y), label, fill=color)


def encode_jpg(img_pil):
    """Encode a PIL image to JPEG bytes."""
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ============================================================
# FEATURE FUNCTIONS (no cv2)
# ============================================================


def fft_features(img_rgb_np):
    """Compute FFT magnitude spectrum from an RGB numpy array."""
    gray = rgb_to_gray(img_rgb_np)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1.0)
    mag = pil_resize(mag, IMG_SIZE, IMG_SIZE)
    mag = np.repeat(mag[np.newaxis, ...], 3, axis=0)
    return torch.tensor(mag, dtype=torch.float32)


def noise_residual(img_rgb_np):
    """Compute noise residual (image minus Gaussian blur)."""
    blur = gaussian_blur(img_rgb_np, radius=2)
    residual = img_rgb_np.astype(np.float32) - blur
    residual = pil_resize(residual.astype(np.uint8), IMG_SIZE, IMG_SIZE)
    residual = torch.tensor(residual).permute(2, 0, 1) / 255.0
    return residual.float()


# ============================================================
# MODEL ARCHITECTURE
# ============================================================


class NanoBananaDetector(nn.Module):
    def __init__(
        self,
        rgb_backbone="convnext_tiny",
        rgb_features=768,
        use_branch_bn=True,
        classifier_linear_shapes=None,
        classifier_bn_after_first=True,
    ):
        super().__init__()
        self.rgb_net = timm.create_model(rgb_backbone, pretrained=False, num_classes=0)
        self.fft_net = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.noise_net = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.use_branch_bn = use_branch_bn

        if use_branch_bn:
            self.rgb_bn = nn.BatchNorm1d(rgb_features)
            self.fft_bn = nn.BatchNorm1d(512)
            self.noise_bn = nn.BatchNorm1d(512)

        if classifier_linear_shapes is None:
            classifier_linear_shapes = {
                0: (rgb_features + 512 + 512, 512),
                4: (512, 128),
                7: (128, 1),
            }

        self.classifier = self._build_classifier(
            classifier_linear_shapes,
            classifier_bn_after_first,
        )

    @staticmethod
    def _build_classifier(linear_shapes, bn_after_first):
        linear_indices = sorted(linear_shapes)

        if linear_indices == [0, 3]:
            in_0, out_0 = linear_shapes[0]
            in_3, out_3 = linear_shapes[3]
            return nn.Sequential(
                nn.Linear(in_0, out_0),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(in_3, out_3),
            )

        if linear_indices == [0, 4, 7]:
            in_0, out_0 = linear_shapes[0]
            in_4, out_4 = linear_shapes[4]
            in_7, out_7 = linear_shapes[7]
            norm_or_identity = nn.BatchNorm1d(out_0) if bn_after_first else nn.Identity()
            return nn.Sequential(
                nn.Linear(in_0, out_0),
                norm_or_identity,
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(in_4, out_4),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(in_7, out_7),
            )

        raise RuntimeError(
            f"Unsupported classifier layout in checkpoint: linear layers {linear_indices}"
        )

    def forward(self, rgb, fft, noise):
        f1 = self.rgb_net(rgb)
        f2 = self.fft_net(fft)
        f3 = self.noise_net(noise)

        if self.use_branch_bn:
            f1 = self.rgb_bn(f1)
            f2 = self.fft_bn(f2)
            f3 = self.noise_bn(f3)

        feats = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(feats).squeeze(1)


def normalize_state_dict(checkpoint):
    """Handle raw state_dicts and common checkpoint wrapper formats."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Classifier checkpoint is not a valid state_dict.")

    normalized = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    return normalized


def infer_detector_config(state_dict):
    """
    Infer the detector architecture from the checkpoint.

    The currently published flux-classifier checkpoint uses ConvNeXt Tiny
    dimensions, branch BatchNorm layers, and a deeper classifier head.
    """
    stem_weight = state_dict.get("rgb_net.stem.0.weight")
    if stem_weight is None:
        raise RuntimeError("Checkpoint is missing rgb_net.stem.0.weight.")

    stem_width = int(stem_weight.shape[0])
    if stem_width == 96:
        rgb_backbone = "convnext_tiny"
        rgb_features = 768
    elif stem_width == 128:
        rgb_backbone = "convnext_base"
        rgb_features = 1024
    else:
        raise RuntimeError(
            f"Unsupported ConvNeXt stem width {stem_width}; cannot infer RGB backbone."
        )

    classifier_linear_shapes = {}
    for key, value in state_dict.items():
        if not key.startswith("classifier.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) != 3 or not parts[1].isdigit() or value.ndim != 2:
            continue
        out_features, in_features = value.shape
        classifier_linear_shapes[int(parts[1])] = (
            int(in_features),
            int(out_features),
        )

    return {
        "rgb_backbone": rgb_backbone,
        "rgb_features": rgb_features,
        "use_branch_bn": "rgb_bn.weight" in state_dict,
        "classifier_linear_shapes": classifier_linear_shapes,
        "classifier_bn_after_first": "classifier.1.weight" in state_dict,
    }


# ============================================================
# DOWNLOAD HELPER
# ============================================================


def download_from_hf(url, dest_path, label, progress_callback=None):
    """Download a model from HuggingFace if not already present."""
    if os.path.exists(dest_path):
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, stream=True, headers=headers, allow_redirects=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {label} (HTTP {r.status_code}).")

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total:
                progress_callback(downloaded / total, downloaded / 1e6)


# ============================================================
# MODEL LOADERS
# ============================================================


def load_yolo_model(progress_callback=None):
    from ultralytics import YOLO

    download_from_hf(YOLO_HF_URL, YOLO_MODEL_PATH, "YOLO11", progress_callback)
    return YOLO(YOLO_MODEL_PATH)


def load_ai_detector(progress_callback=None):
    """Download and load NanoBananaDetector from HuggingFace."""
    download_from_hf(CLASSIFIER_HF_URL, AI_MODEL_PATH, "Flux Classifier", progress_callback)
    checkpoint = torch.load(AI_MODEL_PATH, map_location=DEVICE)
    state_dict = normalize_state_dict(checkpoint)
    model = NanoBananaDetector(**infer_detector_config(state_dict)).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================
# AI DETECTION
# ============================================================


def detect_ai(image_pil, ai_model, threshold=AI_THRESHOLD):
    """
    Run AI classifier on a PIL image.

    This works for both object crops and the full image. The model input is
    resized internally to IMG_SIZE by rgb_transform and the feature functions.
    """
    image_rgb = image_pil.convert("RGB")
    img_np = np.array(image_rgb)

    rgb = rgb_transform(image_rgb).unsqueeze(0).to(DEVICE)
    fft = fft_features(img_np).unsqueeze(0).to(DEVICE)
    noise = noise_residual(img_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = ai_model(rgb, fft, noise)
        prob = torch.sigmoid(logit).item()

    return {"score": prob, "ai_like": prob > threshold}


def empty_ai_result():
    return {"score": 0.0, "ai_like": False}


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_detection(img_pil, yolo_model, ai_model, conf_threshold=0.25):
    """
    Run YOLO detection plus AI classification on a PIL image.
    """
    img_resized = img_pil.convert("RGB").resize(
        (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE),
        Image.BILINEAR,
    )
    img_np = np.array(img_resized)

    if ai_model is not None:
        full_image_ai = detect_ai(img_resized, ai_model)
    else:
        full_image_ai = empty_ai_result()

    results = yolo_model(img_np, conf=conf_threshold, verbose=False)[0]

    detections_list = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = yolo_model.names.get(class_id, f"Class {class_id}")
        detections_list.append({
            "index": i,
            "bbox": [x1, y1, x2, y2],
            "class_id": class_id,
            "class_name": class_name,
            "score": score,
        })

    annotated_pil = img_resized.copy()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for det in detections_list:
            x1, y1, x2, y2 = det["bbox"]

            crop_pil = img_resized.crop((
                max(x1, 0),
                max(y1, 0),
                min(x2, YOLO_IMAGE_SIZE),
                min(y2, YOLO_IMAGE_SIZE),
            ))

            if crop_pil.width == 0 or crop_pil.height == 0:
                det["ai_score"] = 0.0
                det["ai_like"] = False
                det["crop_img"] = None
                continue

            safe_class_name = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in det["class_name"]
            )
            zip_name = (
                f"crop_{det['index']}_{safe_class_name}_"
                f"{int(det['score'] * 100)}.jpg"
            )
            zipf.writestr(zip_name, encode_jpg(crop_pil))
            det["zip_name"] = zip_name
            det["crop_img"] = crop_pil

            if ai_model is not None:
                ai_result = detect_ai(crop_pil, ai_model)
            else:
                ai_result = empty_ai_result()

            det["ai_score"] = ai_result["score"]
            det["ai_like"] = ai_result["ai_like"]

            color = "#ff4444" if ai_result["ai_like"] else "#44cc88"
            label = f"{det['class_name']} {det['score']:.2f} | AI:{det['ai_score']:.2f}"
            draw_box_and_label(annotated_pil, x1, y1, x2, y2, label, color)

    zip_buffer.seek(0)
    return detections_list, annotated_pil, zip_buffer, full_image_ai
