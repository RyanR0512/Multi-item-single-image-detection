import os
import io
import cv2
import torch
import timm
import numpy as np
import zipfile
import requests
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
YOLO_HF_URL   = "https://huggingface.co/RyanR0512/Yolov11-detector/resolve/main/yolo11_detector.pt"
YOLO_MODEL_PATH = "yolo11_detector.pt"
IMG_SIZE      = 224
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TRANSFORMS
# ============================================================
rgb_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# FEATURE FUNCTIONS
# ============================================================
def fft_features(img_rgb):
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f      = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag    = np.log(np.abs(fshift) + 1.0)
    mag    = cv2.resize(mag, (IMG_SIZE, IMG_SIZE))
    mag    = np.repeat(mag[np.newaxis, ...], 3, axis=0)
    return torch.tensor(mag, dtype=torch.float32)

def noise_residual(img_rgb):
    blur     = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    residual = img_rgb.astype(np.float32) - blur.astype(np.float32)
    residual = cv2.resize(residual, (IMG_SIZE, IMG_SIZE))
    residual = torch.tensor(residual).permute(2, 0, 1) / 255.0
    return residual.float()

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class NanoBananaDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_net   = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.fft_net   = timm.create_model("resnet18",      pretrained=False, num_classes=0)
        self.noise_net = timm.create_model("resnet18",      pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 512 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, rgb, fft, noise):
        f1    = self.rgb_net(rgb)
        f2    = self.fft_net(fft)
        f3    = self.noise_net(noise)
        feats = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(feats).squeeze(1)

# ============================================================
# MODEL LOADERS
# ============================================================
def download_yolo_if_needed(progress_callback=None):
    """Download YOLO model from HuggingFace if not present locally."""
    if os.path.exists(YOLO_MODEL_PATH):
        return
    headers    = {"User-Agent": "Mozilla/5.0"}
    r          = requests.get(YOLO_HF_URL, stream=True, headers=headers, allow_redirects=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download YOLO model (HTTP {r.status_code}). Check the URL.")
    total      = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(YOLO_MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total:
                progress_callback(downloaded / total, downloaded / 1e6)


def load_yolo_model():
    from ultralytics import YOLO
    download_yolo_if_needed()
    return YOLO(YOLO_MODEL_PATH)


def load_ai_detector(model_path):
    """Load NanoBananaDetector from an uploaded .pt file path. Returns None if path invalid."""
    if not model_path or not os.path.exists(model_path):
        return None
    model = NanoBananaDetector().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# ============================================================
# AI DETECTION ON A SINGLE CROP
# ============================================================
def detect_ai(crop_bgr, ai_model):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_np  = np.array(img_pil)

    rgb   = rgb_transform(img_pil).unsqueeze(0).to(DEVICE)
    fft   = fft_features(img_np).unsqueeze(0).to(DEVICE)
    noise = noise_residual(img_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = ai_model(rgb, fft, noise)
        prob  = torch.sigmoid(logit).item()

    return {"score": prob, "ai_like": prob > 0.5}

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_detection(img_pil, yolo_model, ai_model, conf_threshold=0.25):
    """
    Run YOLO detection + AI classification on a PIL image.

    Args:
        img_pil         : PIL.Image  — the uploaded image
        yolo_model      : loaded YOLO model (from load_yolo_model)
        ai_model        : loaded NanoBananaDetector or None
        conf_threshold  : float — YOLO confidence cutoff

    Returns:
        detections_list : list of dicts with keys:
                          index, bbox, class_id, class_name, score,
                          ai_score, ai_like, crop_img, zip_name
        annotated_pil   : PIL.Image — annotated result image
        zip_buffer      : BytesIO  — ZIP of cropped detections
    """
    img_rgb     = np.array(img_pil.convert("RGB"))
    img_bgr     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (640, 640))

    results = yolo_model(img_resized, conf=conf_threshold, verbose=False)[0]

    detections_list = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score      = float(box.conf[0])
        class_id   = int(box.cls[0])
        class_name = yolo_model.names.get(class_id, f"Class {class_id}")
        detections_list.append({
            "index":      i,
            "bbox":       [x1, y1, x2, y2],
            "class_id":   class_id,
            "class_name": class_name,
            "score":      score,
        })

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for det in detections_list:
            x1, y1, x2, y2 = det["bbox"]
            crop = img_resized[max(y1, 0):y2, max(x1, 0):x2]

            if crop.size == 0:
                det["ai_score"] = 0.0
                det["ai_like"]  = False
                det["crop_img"] = None
                continue

            # Save crop to ZIP
            ok, buffer = cv2.imencode(".jpg", crop)
            zip_name   = f"crop_{det['index']}_{det['class_name']}_{int(det['score']*100)}.jpg"
            zipf.writestr(zip_name, buffer.tobytes())
            det["zip_name"] = zip_name

            # Store crop as RGB PIL for display
            det["crop_img"] = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # AI detection
            if ai_model is not None:
                ai_result = detect_ai(crop, ai_model)
            else:
                ai_result = {"score": 0.0, "ai_like": False}

            det["ai_score"] = ai_result["score"]
            det["ai_like"]  = ai_result["ai_like"]

            # Annotate image
            color = (0, 0, 255) if ai_result["ai_like"] else (0, 255, 0)
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['score']:.2f} | AI:{det['ai_score']:.2f}"
            cv2.putText(img_resized, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    annotated_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    zip_buffer.seek(0)
    return detections_list, annotated_pil, zip_buffer
