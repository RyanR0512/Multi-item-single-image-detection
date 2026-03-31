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
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Flux AI Detector",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0d0d0d;
        color: #f0ede6;
    }

    /* Header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -1px;
        color: #f0ede6;
        margin: 0;
    }
    .hero-title span {
        color: #f5c542;
    }
    .hero-sub {
        font-size: 1rem;
        color: #888;
        margin-top: 0.4rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Cards */
    .card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #f5c542;
        margin-bottom: 1rem;
    }

    /* Detection result rows */
    .det-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border: 1px solid #2a2a2a;
        background: #111;
    }
    .det-row.ai {
        border-left: 3px solid #ff4444;
    }
    .det-row.real {
        border-left: 3px solid #44cc88;
    }
    .det-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #f0ede6;
    }
    .det-badge {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .badge-ai {
        background: rgba(255, 68, 68, 0.15);
        color: #ff6666;
        border: 1px solid #ff4444;
    }
    .badge-real {
        background: rgba(68, 204, 136, 0.15);
        color: #44cc88;
        border: 1px solid #44cc88;
    }
    .det-scores {
        font-size: 0.75rem;
        color: #666;
        font-family: 'Space Mono', monospace;
    }

    /* Score bar */
    .score-bar-wrap {
        width: 80px;
        height: 4px;
        background: #2a2a2a;
        border-radius: 2px;
        overflow: hidden;
        display: inline-block;
        vertical-align: middle;
        margin-left: 8px;
    }
    .score-bar-fill-ai {
        height: 100%;
        background: #ff4444;
        border-radius: 2px;
    }
    .score-bar-fill-real {
        height: 100%;
        background: #44cc88;
        border-radius: 2px;
    }

    /* Stat boxes */
    .stat-grid {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .stat-box {
        flex: 1;
        background: #111;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-num {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #f5c542;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #666;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    .stat-box.ai-box .stat-num { color: #ff4444; }
    .stat-box.real-box .stat-num { color: #44cc88; }

    /* Upload zone */
    .upload-hint {
        text-align: center;
        color: #555;
        font-size: 0.85rem;
        padding: 1rem 0;
        font-family: 'Space Mono', monospace;
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #2a2a2a;
        margin: 1rem 0;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: #f5c542 !important;
        color: #0d0d0d !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        padding: 0.6rem 2rem !important;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }
    .stSlider label, .stSlider .st-bd {
        color: #888 !important;
    }
    [data-testid="stFileUploader"] {
        background: #1a1a1a !important;
        border: 1px dashed #333 !important;
        border-radius: 12px !important;
    }
    .stProgress > div > div {
        background: #f5c542 !important;
    }
    section[data-testid="stSidebar"] {
        background: #111 !important;
        border-right: 1px solid #2a2a2a !important;
    }
    .stAlert {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIG
# ============================================================
YOLO_HF_URL = "https://huggingface.co/RyanR0512/Yolov11-detector/resolve/main/yolo11_detector.pt"
YOLO_MODEL_PATH = "yolo11n.pt"
AI_MODEL_PATH = "flux_classifier.pt"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1.0)
    mag = cv2.resize(mag, (IMG_SIZE, IMG_SIZE))
    mag = np.repeat(mag[np.newaxis, ...], 3, axis=0)
    return torch.tensor(mag, dtype=torch.float32)

def noise_residual(img_rgb):
    blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
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
        self.rgb_net = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.fft_net = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.noise_net = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 512 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, rgb, fft, noise):
        f1 = self.rgb_net(rgb)
        f2 = self.fft_net(fft)
        f3 = self.noise_net(noise)
        feats = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(feats).squeeze(1)

# ============================================================
# CACHED MODEL LOADERS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    from ultralytics import YOLO
    if not os.path.exists(YOLO_MODEL_PATH):
        with st.spinner("Downloading YOLO11 model from HuggingFace…"):
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(YOLO_HF_URL, stream=True, headers=headers, allow_redirects=True)
            if r.status_code != 200:
                st.error(f"Failed to download YOLO model (HTTP {r.status_code}). Check the URL.")
                st.stop()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            progress = st.progress(0, text="Downloading YOLO11…")
            with open(YOLO_MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.progress(min(downloaded / total, 1.0), text=f"Downloading YOLO11… {downloaded/1e6:.1f} MB")
            progress.empty()
    return YOLO(YOLO_MODEL_PATH)

@st.cache_resource(show_spinner=False)
def load_ai_detector():
    if not os.path.exists(AI_MODEL_PATH):
        return None
    model = NanoBananaDetector().to(DEVICE)
    model.load_state_dict(torch.load(AI_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ============================================================
# AI DETECTION ON A CROP
# ============================================================
def detect_ai(crop_bgr, ai_model):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_np = np.array(img_pil)

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
def run_detection(img_pil, conf_threshold=0.25):
    yolo_model = load_yolo_model()
    ai_model   = load_ai_detector()

    # Convert PIL -> BGR numpy
    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (640, 640))

    results = yolo_model(img_resized, conf=conf_threshold, verbose=False)[0]

    detections_list = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score      = float(box.conf[0])
        class_id   = int(box.cls[0])
        class_name = yolo_model.names.get(class_id, f"Class {class_id}")
        detections_list.append({
            "index": i,
            "bbox": [x1, y1, x2, y2],
            "class_id": class_id,
            "class_name": class_name,
            "score": score,
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

            # ZIP crop
            ok, buffer = cv2.imencode(".jpg", crop)
            zip_name = f"crop_{det['index']}_{det['class_name']}_{int(det['score']*100)}.jpg"
            zipf.writestr(zip_name, buffer.tobytes())
            det["zip_name"] = zip_name

            # Store crop as RGB PIL for display
            det["crop_img"] = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # AI detection
            if ai_model is not None:
                ai_result = detect_ai(crop, ai_model)
            else:
                # Fallback: random-ish mock if no model weights found
                ai_result = {"score": 0.0, "ai_like": False}

            det["ai_score"] = ai_result["score"]
            det["ai_like"]  = ai_result["ai_like"]

            # Annotate
            color = (0, 0, 255) if ai_result["ai_like"] else (0, 255, 0)
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['score']:.2f} | AI:{det['ai_score']:.2f}"
            cv2.putText(img_resized, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    annotated_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    zip_buffer.seek(0)
    return detections_list, annotated_pil, zip_buffer

# ============================================================
# UI LAYOUT
# ============================================================

# --- Hero Header ---
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Nano<span>Banana</span> Detector</div>
    <div class="hero-sub">YOLO11 Object Detection · AI Image Analysis</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="card-title">⚙ Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Minimum YOLO detection confidence to include a result."
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">ℹ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#666; font-size:0.8rem; line-height:1.6;">
    Upload any image to detect objects with <b style="color:#f5c542">YOLO11</b> and analyse 
    each crop with <b style="color:#f5c542">NanoBananaDetector</b> to determine whether 
    detected regions appear AI-generated or real.<br><br>
    🔴 Red box = likely AI-generated<br>
    🟢 Green box = likely real
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    device_label = "🖥 GPU (CUDA)" if DEVICE == "cuda" else "💻 CPU"
    st.markdown(f'<div style="color:#555; font-size:0.75rem; font-family: monospace;">Running on: {device_label}</div>', unsafe_allow_html=True)

    if not os.path.exists(AI_MODEL_PATH):
        st.warning(f"⚠ `{AI_MODEL_PATH}` not found.\nPlace it in the app directory to enable AI detection.")

# --- Main Columns ---
col_left, col_right = st.columns([1, 1.35], gap="large")

with col_left:
    st.markdown('<div class="card-title">📁 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop or browse an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        img_pil = Image.open(uploaded_file).convert("RGB")
        st.image(img_pil, caption="Original Image", use_container_width=True)
        run_btn = st.button("🔍  RUN DETECTION")
    else:
        st.markdown('<div class="upload-hint">↑ upload an image to get started</div>', unsafe_allow_html=True)
        run_btn = False

with col_right:
    if uploaded_file and run_btn:
        with st.spinner("Running detection pipeline…"):
            detections, annotated_img, zip_buf = run_detection(img_pil, conf_threshold=conf_threshold)

        # Store in session state
        st.session_state["detections"]    = detections
        st.session_state["annotated_img"] = annotated_img
        st.session_state["zip_buf"]       = zip_buf

    if "detections" in st.session_state and st.session_state["detections"] is not None:
        detections    = st.session_state["detections"]
        annotated_img = st.session_state["annotated_img"]
        zip_buf       = st.session_state["zip_buf"]

        # Annotated image
        st.markdown('<div class="card-title">🖼 Annotated Result</div>', unsafe_allow_html=True)
        st.image(annotated_img, use_container_width=True)

        # Summary stats
        n_total = len(detections)
        n_ai    = sum(1 for d in detections if d.get("ai_like", False))
        n_real  = n_total - n_ai

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-num">{n_total}</div>
                <div class="stat-label">Detected</div>
            </div>
            <div class="stat-box ai-box">
                <div class="stat-num">{n_ai}</div>
                <div class="stat-label">AI-Generated</div>
            </div>
            <div class="stat-box real-box">
                <div class="stat-num">{n_real}</div>
                <div class="stat-label">Real</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detection rows
        st.markdown('<div class="card-title">📋 Detection Breakdown</div>', unsafe_allow_html=True)

        if not detections:
            st.info("No objects detected above the confidence threshold. Try lowering it in the sidebar.")
        else:
            for det in detections:
                ai_like   = det.get("ai_like", False)
                ai_score  = det.get("ai_score", 0.0)
                yolo_conf = det["score"]
                row_cls   = "ai" if ai_like else "real"
                badge_cls = "badge-ai" if ai_like else "badge-real"
                badge_txt = "AI-GENERATED" if ai_like else "REAL"
                bar_cls   = "score-bar-fill-ai" if ai_like else "score-bar-fill-real"
                bar_pct   = int(ai_score * 100)

                st.markdown(f"""
                <div class="det-row {row_cls}">
                    <div>
                        <div class="det-label">#{det['index']+1} {det['class_name']}</div>
                        <div class="det-scores">
                            YOLO: {yolo_conf:.2f} &nbsp;|&nbsp; AI score: {ai_score:.2f}
                            <span class="score-bar-wrap">
                                <div class="{bar_cls}" style="width:{bar_pct}%"></div>
                            </span>
                        </div>
                    </div>
                    <span class="det-badge {badge_cls}">{badge_txt}</span>
                </div>
                """, unsafe_allow_html=True)

            # Crop gallery
            crops = [d for d in detections if d.get("crop_img") is not None]
            if crops:
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🔬 Cropped Detections</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(crops), 4))
                for i, det in enumerate(crops):
                    color_label = "🔴 AI" if det["ai_like"] else "🟢 Real"
                    with cols[i % 4]:
                        st.image(
                            det["crop_img"],
                            caption=f"#{det['index']+1} {det['class_name']}\n{color_label} ({det['ai_score']:.2f})",
                            use_container_width=True
                        )

            # Download button
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.download_button(
                label="⬇  Download Crops ZIP",
                data=zip_buf,
                file_name="cropped_detections.zip",
                mime="application/zip",
                use_container_width=True,
            )

    elif not uploaded_file:
        st.markdown("""
        <div style="height:200px; display:flex; align-items:center; justify-content:center; color:#333; font-family:monospace; font-size:0.85rem;">
            results will appear here
        </div>
        """, unsafe_allow_html=True)
