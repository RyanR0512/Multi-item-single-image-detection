import os

import subprocess
import streamlit as st

st.text(subprocess.run(["python", "--version"], capture_output=True, text=True).stdout)
st.text(subprocess.run(["pip", "list"], capture_output=True, text=True).stdout)
st.stop()


from PIL import Image
import detection

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

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background-color: #0d0d0d; color: #f0ede6; }

    .hero-header { text-align: center; padding: 2.5rem 1rem 1.5rem; }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.6rem; font-weight: 700;
        letter-spacing: -1px; color: #f0ede6; margin: 0;
    }
    .hero-title span { color: #f5c542; }
    .hero-sub { font-size: 1rem; color: #888; margin-top: 0.4rem; font-weight: 300; letter-spacing: 0.5px; }

    .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .card-title {
        font-family: 'Space Mono', monospace; font-size: 0.75rem;
        letter-spacing: 2px; text-transform: uppercase; color: #f5c542; margin-bottom: 1rem;
    }

    .det-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;
        border: 1px solid #2a2a2a; background: #111;
    }
    .det-row.ai  { border-left: 3px solid #ff4444; }
    .det-row.real { border-left: 3px solid #44cc88; }
    .det-label { font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #f0ede6; }
    .det-badge { font-family: 'Space Mono', monospace; font-size: 0.7rem; padding: 3px 10px; border-radius: 20px; font-weight: 700; letter-spacing: 1px; }
    .badge-ai   { background: rgba(255,68,68,0.15); color: #ff6666; border: 1px solid #ff4444; }
    .badge-real { background: rgba(68,204,136,0.15); color: #44cc88; border: 1px solid #44cc88; }
    .det-scores { font-size: 0.75rem; color: #666; font-family: 'Space Mono', monospace; }

    .score-bar-wrap { width: 80px; height: 4px; background: #2a2a2a; border-radius: 2px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 8px; }
    .score-bar-fill-ai   { height: 100%; background: #ff4444; border-radius: 2px; }
    .score-bar-fill-real { height: 100%; background: #44cc88; border-radius: 2px; }

    .stat-grid { display: flex; gap: 0.75rem; margin-bottom: 1rem; }
    .stat-box { flex: 1; background: #111; border: 1px solid #2a2a2a; border-radius: 10px; padding: 1rem; text-align: center; }
    .stat-num  { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #f5c542; line-height: 1; }
    .stat-label { font-size: 0.7rem; color: #666; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.3rem; }
    .stat-box.ai-box   .stat-num { color: #ff4444; }
    .stat-box.real-box .stat-num { color: #44cc88; }

    .upload-hint { text-align: center; color: #555; font-size: 0.85rem; padding: 1rem 0; font-family: 'Space Mono', monospace; }
    .divider { border: none; border-top: 1px solid #2a2a2a; margin: 1rem 0; }

    .stButton > button {
        background: #f5c542 !important; color: #0d0d0d !important;
        border: none !important; border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important; font-weight: 700 !important;
        font-size: 0.85rem !important; letter-spacing: 1px !important;
        padding: 0.6rem 2rem !important; width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    [data-testid="stFileUploader"] { background: #1a1a1a !important; border: 1px dashed #333 !important; border-radius: 12px !important; }
    .stProgress > div > div { background: #f5c542 !important; }
    section[data-testid="stSidebar"] { background: #111 !important; border-right: 1px solid #2a2a2a !important; }
    .stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CACHED MODEL LOADERS (live here so st.cache_resource works)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_yolo_model():
    progress_bar = st.progress(0, text="Downloading YOLO11 model…")
    def cb(frac, mb):
        progress_bar.progress(frac, text=f"Downloading YOLO11… {mb:.1f} MB")
    detection.download_yolo_if_needed(progress_callback=cb)
    progress_bar.empty()
    return detection.load_yolo_model()

@st.cache_resource(show_spinner=False)
def get_ai_model(model_path):
    return detection.load_ai_detector(model_path)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Nano<span>Banana</span> Detector</div>
    <div class="hero-sub">YOLO11 Object Detection · AI Image Analysis</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="card-title">⚙ Settings</div>', unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Minimum YOLO detection confidence to include a result."
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧠 AI Classifier Model</div>', unsafe_allow_html=True)

    # --- User uploads the classifier weights (like the TFLite app) ---
    uploaded_model = st.file_uploader(
        "Upload flux_classifier.pt",
        type=["pt"],
        help="Upload your NanoBananaDetector weights (.pt file)."
    )

    ai_model_path = None
    if uploaded_model is not None:
        ai_model_path = "temp_classifier.pt"
        with open(ai_model_path, "wb") as f:
            f.write(uploaded_model.read())
        st.success(f"✅ Classifier loaded: {uploaded_model.name}")
    else:
        st.info("ℹ Upload a `.pt` classifier to enable AI detection.\nYOLO detection still works without it.")

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
    device_label = "🖥 GPU (CUDA)" if detection.DEVICE == "cuda" else "💻 CPU"
    st.markdown(f'<div style="color:#555; font-size:0.75rem; font-family:monospace;">Running on: {device_label}</div>', unsafe_allow_html=True)

# ============================================================
# MAIN COLUMNS
# ============================================================
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
        try:
            with st.spinner("Loading YOLO model…"):
                yolo_model = get_yolo_model()

            ai_model = get_ai_model(ai_model_path) if ai_model_path else None

            with st.spinner("Running detection pipeline…"):
                detections, annotated_img, zip_buf = detection.run_detection(
                    img_pil, yolo_model, ai_model, conf_threshold=conf_threshold
                )

            st.session_state["detections"]    = detections
            st.session_state["annotated_img"] = annotated_img
            st.session_state["zip_buf"]       = zip_buf

        except Exception as e:
            st.error(f"⚠️ Error during detection: {e}")

    if "detections" in st.session_state and st.session_state["detections"] is not None:
        detections    = st.session_state["detections"]
        annotated_img = st.session_state["annotated_img"]
        zip_buf       = st.session_state["zip_buf"]

        st.markdown('<div class="card-title">🖼 Annotated Result</div>', unsafe_allow_html=True)
        st.image(annotated_img, use_container_width=True)

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
        <div style="height:200px; display:flex; align-items:center; justify-content:center;
                    color:#333; font-family:monospace; font-size:0.85rem;">
            results will appear here
        </div>
        """, unsafe_allow_html=True)
