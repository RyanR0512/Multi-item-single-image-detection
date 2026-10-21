import os
import io
import zipfile

import cv2  # already installed as a dependency of ultralytics
import numpy as np
import requests
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms

# ============================================================
# CONFIG
# ============================================================
YOLO_HF_URL = "https://huggingface.co/RyanR0512/Yolov11-detector/resolve/main/best.pt"
CLASSIFIER_HF_URL = "https://huggingface.co/RyanR0512/flux-classifier/resolve/main/best_f1_state_dict.pt"
YOLO_MODEL_PATH = "yolo11_detector.pt"
AI_MODEL_PATH = "flux_classifier.pt"
IMG_SIZE = 224
YOLO_IMAGE_SIZE = 640
AI_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Activation used in the classifier head. trainPyTorch_v2.py uses GELU.
# If best_model.pt came from an older trainer that used ReLU, set this to "relu".
# Activations have no weights, so the checkpoint loads either way; a wrong
# choice silently changes the outputs. check_parity.py will tell you which is right.
CLASSIFIER_ACTIVATION = "gelu"

# ============================================================
# TRANSFORMS (identical to val_tf in trainPyTorch_v2.py)
# ============================================================
rgb_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
# FEATURE FUNCTIONS
# Copied verbatim from trainPyTorch_v2.py so training and inference
# see exactly the same inputs. If you change one, change both
# (or better, move these into a shared features.py both files import).
# ============================================================


def fft_features(img_rgb: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    freq = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(freq) + 1.0)
    mag = cv2.resize(mag, (img_size, img_size))
    mag = (mag - mag.mean()) / (mag.std() + 1e-8)
    mag = np.repeat(mag[np.newaxis, ...], 3, axis=0)
    return torch.tensor(mag, dtype=torch.float32)


def noise_residual(img_rgb: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    residual = img_rgb.astype(np.float32) - blur.astype(np.float32)
    residual = cv2.resize(residual, (img_size, img_size)) / 127.5
    return torch.tensor(residual).permute(2, 0, 1).float()


# ============================================================
# DRAWING / ENCODING HELPERS
# ============================================================


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
# MODEL ARCHITECTURE
# ============================================================

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU}


class NanoBananaDetector(nn.Module):
    def __init__(
        self,
        rgb_backbone="convnext_tiny",
        rgb_features=768,
        use_branch_bn=True,
        classifier_linear_shapes=None,
        classifier_bn_after_first=True,
        activation=CLASSIFIER_ACTIVATION,
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

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; use one of {list(ACTIVATIONS)}")

        self.classifier = self._build_classifier(
            classifier_linear_shapes,
            classifier_bn_after_first,
            ACTIVATIONS[activation],
        )

    @staticmethod
    def _build_classifier(linear_shapes, bn_after_first, act):
        linear_indices = sorted(linear_shapes)

        if linear_indices == [0, 3]:
            in_0, out_0 = linear_shapes[0]
            in_3, out_3 = linear_shapes[3]
            return nn.Sequential(
                nn.Linear(in_0, out_0),
                act(),
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
                act(),
                nn.Dropout(0.4),
                nn.Linear(in_4, out_4),
                act(),
                nn.Dropout(0.2),
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


def unwrap_checkpoint(checkpoint):
    """Return (state_dict, metadata). Metadata is {} for raw state_dicts."""
    metadata = {}
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                metadata = {k: v for k, v in checkpoint.items() if k != key}
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Classifier checkpoint is not a valid state_dict.")

    normalized = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    return normalized, metadata


def infer_detector_config(state_dict, metadata):
    """Infer the detector architecture from the checkpoint."""
    stem_weight = state_dict.get("rgb_net.stem.0.weight")
    if stem_weight is None:
        raise RuntimeError("Checkpoint is missing rgb_net.stem.0.weight.")

    stem_width = int(stem_weight.shape[0])
    if stem_width == 96:
        rgb_backbone, rgb_features = "convnext_tiny", 768
    elif stem_width == 128:
        rgb_backbone, rgb_features = "convnext_base", 1024
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
        classifier_linear_shapes[int(parts[1])] = (int(in_features), int(out_features))

    # A full checkpoint saved by trainPyTorch_v2.py carries "args"; that trainer uses GELU.
    activation = "gelu" if "args" in metadata else CLASSIFIER_ACTIVATION

    return {
        "rgb_backbone": rgb_backbone,
        "rgb_features": rgb_features,
        "use_branch_bn": "rgb_bn.weight" in state_dict,
        "classifier_linear_shapes": classifier_linear_shapes,
        "classifier_bn_after_first": "classifier.1.weight" in state_dict,
        "activation": activation,
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
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total:
                progress_callback(downloaded / total, downloaded / 1e6)
    os.replace(tmp_path, dest_path)  # avoid leaving a half-downloaded model behind


# ============================================================
# MODEL LOADERS
# ============================================================


def load_yolo_model(progress_callback=None):
    from ultralytics import YOLO

    download_from_hf(YOLO_HF_URL, YOLO_MODEL_PATH, "YOLO11", progress_callback)
    return YOLO(YOLO_MODEL_PATH)


def load_ai_detector(progress_callback=None, path=None):
    """Download and load NanoBananaDetector. Pass path= to load a local checkpoint."""
    if path is None:
        download_from_hf(CLASSIFIER_HF_URL, AI_MODEL_PATH, "Flux Classifier", progress_callback)
        path = AI_MODEL_PATH

    # Your own checkpoint; full v2 checkpoints hold optimizer/scheduler state too.
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict, metadata = unwrap_checkpoint(checkpoint)
    config = infer_detector_config(state_dict, metadata)

    model = NanoBananaDetector(**config).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Use the threshold the model was trained/evaluated with, if saved.
    model.threshold = float(metadata.get("threshold", AI_THRESHOLD))
    model.config = config
    return model


# ============================================================
# AI DETECTION
# ============================================================


def detect_ai(image_pil, ai_model, threshold=None):
    """Run the AI classifier on a PIL image (object crop or full image)."""
    if threshold is None:
        threshold = getattr(ai_model, "threshold", AI_THRESHOLD)

    image_rgb = image_pil.convert("RGB")
    img_np = np.array(image_rgb)  # uint8 RGB, same as FluxDataset.__getitem__

    rgb = rgb_transform(image_rgb).unsqueeze(0).to(DEVICE)
    fft = fft_features(img_np).unsqueeze(0).to(DEVICE)
    noise = noise_residual(img_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = ai_model(rgb, fft, noise)
        prob = torch.sigmoid(logit).item()

    # >= matches the Meter class in the trainer
    return {"score": prob, "ai_like": prob >= threshold}


def empty_ai_result():
    return {"score": 0.0, "ai_like": False}


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_detection(img_pil, yolo_model, ai_model, conf_threshold=0.25):
    """
    Run YOLO detection plus AI classification on a PIL image.

    Returns (detections, annotated_pil, zip_buffer, full_image_ai).
    """
    img_resized = img_pil.convert("RGB").resize(
        (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE),
        Image.BILINEAR,
    )
    img_np = np.array(img_resized)

    full_image_ai = detect_ai(img_resized, ai_model) if ai_model is not None else empty_ai_result()

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
            zip_name = f"crop_{det['index']}_{safe_class_name}_{int(det['score'] * 100)}.jpg"
            zipf.writestr(zip_name, encode_jpg(crop_pil))
            det["zip_name"] = zip_name
            det["crop_img"] = crop_pil

            ai_result = detect_ai(crop_pil, ai_model) if ai_model is not None else empty_ai_result()
            det["ai_score"] = ai_result["score"]
            det["ai_like"] = ai_result["ai_like"]

            color = "#ff4444" if ai_result["ai_like"] else "#44cc88"
            label = f"{det['class_name']} {det['score']:.2f} | AI:{det['ai_score']:.2f}"
            draw_box_and_label(annotated_pil, x1, y1, x2, y2, label, color)

    zip_buffer.seek(0)
    return detections_list, annotated_pil, zip_buffer, full_image_ai
