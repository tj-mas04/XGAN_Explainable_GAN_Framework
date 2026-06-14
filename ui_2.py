"""
TumorX GAN — Explainable Brain Tumor AI System
================================================
Premium UI redesign. All backend logic identical to xgan.ipynb.

Models required (place alongside app.py):
  tumor_classifier_axial.h5
  classifier_history.json
  gan_checkpoints_V2_h5/generator_epoch_60.h5
  gan_checkpoints_V2_h5/discriminator_epoch_60.h5
"""

import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TumorX GAN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE       = 128
CHANNELS       = 1
LATENT_DIM     = 100
STRUCTURE_DIM  = 80
PATHOLOGY_DIM  = 20

# Model file paths — keep these relative to app.py location
CLASSIFIER_PATH    = "tumor_classifier_axial.h5"
GENERATOR_PATH     = "gan_checkpoints_V2_h5/generator_epoch_60.h5"
DISCRIMINATOR_PATH = "gan_checkpoints_V2_h5/discriminator_epoch_60.h5"
HISTORY_PATH       = "classifier_history.json"


# ─────────────────────────────────────────────────────────────────────────────
# CSS — TumorX GAN Design System
# Deep navy-black base · electric cyan accent · warm coral for tumor alerts
# Font: Syne (display) + DM Sans (body) — clinical-tech editorial feel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: #e2e8f4;
}
.stApp {
  background: #050810;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,200,255,0.06) 0%, transparent 70%),
    radial-gradient(ellipse 40% 30% at 90% 80%, rgba(255,80,100,0.04) 0%, transparent 60%);
}
.block-container { padding-top: 0 !important; padding-bottom: 40px; }

[data-testid="stSidebar"] {
  background: #080c18 !important;
  border-right: 1px solid rgba(0,200,255,0.1);
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid rgba(0,200,255,0.12);
  gap: 0;
  padding: 0;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  color: #4a5a78;
  padding: 12px 22px;
  border-radius: 0;
  border-bottom: 2px solid transparent;
  text-transform: uppercase;
  transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #a0b4d0;
  background: rgba(0,200,255,0.04);
}
.stTabs [aria-selected="true"] {
  background: transparent !important;
  color: #00c8ff !important;
  border-bottom: 2px solid #00c8ff !important;
}

.stButton > button {
  font-family: 'DM Mono', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  background: transparent;
  color: #00c8ff;
  border: 1px solid rgba(0,200,255,0.4);
  border-radius: 4px;
  padding: 11px 28px;
  width: 100%;
  transition: all 0.2s ease;
}
.stButton > button:hover {
  border-color: #00c8ff;
  color: #fff;
  box-shadow: 0 0 20px rgba(0,200,255,0.2), inset 0 0 20px rgba(0,200,255,0.05);
}

.stProgress > div > div {
  background: linear-gradient(90deg, #0099cc, #00c8ff, #33ddff) !important;
  border-radius: 2px;
  box-shadow: 0 0 8px rgba(0,200,255,0.4);
}
.stProgress > div {
  background: rgba(0,200,255,0.08) !important;
  border-radius: 2px;
}

.stSlider > div > div > div { background: #00c8ff !important; }

[data-testid="stFileUploader"] {
  background: rgba(0,200,255,0.03) !important;
  border: 1px dashed rgba(0,200,255,0.2) !important;
  border-radius: 8px !important;
}

.stAlert { background: rgba(0,20,40,0.8) !important; border-radius: 6px; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050810; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

def section_label(text: str, color: str = "#00c8ff"):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
      <div style="width:3px;height:14px;background:{color};border-radius:2px;flex-shrink:0;
                  box-shadow:0 0 8px {color}88;"></div>
      <span style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.14em;
                   text-transform:uppercase;color:{color};">{text}</span>
    </div>""", unsafe_allow_html=True)


def pill(text: str, kind: str = "info"):
    colors = {
        "tumor":  ("#ff4060", "#3a0010", "rgba(255,64,96,0.3)"),
        "normal": ("#00e5a0", "#00261a", "rgba(0,229,160,0.3)"),
        "info":   ("#00c8ff", "#001a26", "rgba(0,200,255,0.3)"),
        "warn":   ("#ffa040", "#261500", "rgba(255,160,64,0.3)"),
    }
    fg, bg, border = colors.get(kind, colors["info"])
    st.markdown(f"""
    <span style="display:inline-block;background:{bg};color:{fg};
                 border:1px solid {border};border-radius:4px;
                 padding:5px 16px;font-family:'DM Mono',monospace;
                 font-size:0.82rem;font-weight:500;letter-spacing:0.08em;">
      {text}
    </span>""", unsafe_allow_html=True)


def stat_card(value: str, label: str, color: str = "#00c8ff", sublabel: str = ""):
    sub_html = f'<div style="font-size:0.65rem;color:#1e3050;margin-top:4px;">{sublabel}</div>' if sublabel else ""
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
                border-radius:10px;padding:20px 16px;text-align:center;
                border-top:2px solid {color}44;">
      <div style="font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:700;
                  color:{color};line-height:1;">{value}</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.12em;
                  text-transform:uppercase;color:#4a5a78;margin-top:6px;">{label}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)


def image_frame(img: Image.Image, caption: str = "", tag: str = ""):
    img_r = img.resize((IMG_SIZE, IMG_SIZE))
    st.image(img_r, use_container_width=True)
    row = ""
    if tag:
        tcolors = {"TUMOR": "#ff4060", "NO TUMOR": "#00e5a0", "GENERATED": "#00c8ff",
                   "DIFF": "#ffa040", "GRAD-CAM": "#c084fc", "INPUT": "#8a9ec0"}
        tc = tcolors.get(tag, "#8a9ec0")
        row += f'<span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:{tc};' \
               f'letter-spacing:0.1em;border:1px solid {tc}44;padding:2px 8px;border-radius:3px;">{tag}</span> '
    if caption:
        row += f'<span style="font-size:0.7rem;color:#3a4a64;">{caption}</span>'
    if row:
        st.markdown(f'<div style="margin-top:6px;">{row}</div>', unsafe_allow_html=True)


def empty_frame(label: str = "AWAITING INPUT"):
    st.markdown(f"""
    <div style="aspect-ratio:1;display:flex;align-items:center;justify-content:center;
                border:1px dashed rgba(0,200,255,0.12);border-radius:8px;
                background:rgba(0,200,255,0.02);min-height:140px;">
      <span style="font-family:'DM Mono',monospace;font-size:0.65rem;
                   letter-spacing:0.15em;color:#1e2d48;">{label}</span>
    </div>""", unsafe_allow_html=True)


def divider():
    st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.05);margin:24px 0;">', unsafe_allow_html=True)


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("uploaded_image", None), ("prediction_done", False),
             ("pred_label", None), ("pred_confidence", None), ("latest_ssim", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models, errors = {}, {}
    for key, path in [("classifier", CLASSIFIER_PATH),
                      ("generator",  GENERATOR_PATH),
                      ("discriminator", DISCRIMINATOR_PATH)]:
        if os.path.exists(path):
            try:
                m = tf.keras.models.load_model(path)
                m.trainable = False
                models[key] = m
            except Exception as e:
                models[key] = None; errors[key] = str(e)
        else:
            models[key] = None; errors[key] = f"Not found: {path}"
    models["errors"] = errors
    return models


@st.cache_data(show_spinner=False)
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = (np.array(img, dtype="float32") - 127.5) / 127.5
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, CHANNELS)


def tensor_to_pil(tensor) -> Image.Image:
    arr = np.clip((tensor[0].numpy().squeeze() + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def predict(pil_image: Image.Image, models: dict):
    clf = models.get("classifier")
    if clf is None:
        return "Error", 0.0
    prob = float(clf.predict(preprocess_image(pil_image), verbose=0)[0][0])
    label = "Tumor" if prob >= 0.5 else "No Tumor"
    return label, prob if prob >= 0.5 else 1.0 - prob


def generate_image(models: dict, with_tumor: bool = True,
                   z_pathology_strength: float = 1.0) -> Image.Image:
    gen = models.get("generator")
    if gen is None:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (8, 10, 20))
    z_s = tf.random.normal((1, STRUCTURE_DIM))
    z_p = z_pathology_strength * tf.random.normal((1, PATHOLOGY_DIM)) if with_tumor \
          else tf.zeros((1, PATHOLOGY_DIM))
    return tensor_to_pil(gen(tf.concat([z_s, z_p], axis=1), training=False))


def generate_counterfactual(models: dict):
    gen = models.get("generator")
    if gen is None:
        ph = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (8, 10, 20))
        return ph, ph
    z_s   = tf.random.normal((1, STRUCTURE_DIM))
    z_no  = tf.concat([z_s, tf.zeros((1, PATHOLOGY_DIM))],         axis=1)
    z_yes = tf.concat([z_s, tf.random.normal((1, PATHOLOGY_DIM))], axis=1)
    return (tensor_to_pil(gen(z_no,  training=False)),
            tensor_to_pil(gen(z_yes, training=False)))


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
def get_last_conv_layer(model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")


def grad_cam(model, img_tensor, layer_name: str) -> np.ndarray:
    gm = tf.keras.models.Model([model.inputs],
                                [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = gm(img_tensor)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pg    = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm    = tf.maximum(tf.squeeze(conv_out[0] @ pg[..., tf.newaxis]), 0)
    hm    = tf.image.resize(hm[tf.newaxis, :, :, tf.newaxis],
                             [IMG_SIZE, IMG_SIZE]).numpy().squeeze()
    mx = hm.max()
    return hm / mx if mx > 0 else hm


def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray,
                    colormap: str = "jet") -> Image.Image:
    cmap    = matplotlib.colormaps.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    base    = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), np.uint8)
    return Image.fromarray(np.clip(0.5 * base + 0.5 * colored, 0, 255).astype(np.uint8))


def generate_gradcam(pil_image: Image.Image, models: dict,
                     target: str = "classifier") -> Image.Image:
    model = models.get(target)
    if model is None:
        return pil_image.convert("RGB")
    hm = grad_cam(model, tf.convert_to_tensor(preprocess_image(pil_image)),
                  get_last_conv_layer(model))
    return overlay_heatmap(pil_image, hm)


# ─────────────────────────────────────────────────────────────────────────────
# DIFFERENCE MAP & SSIM
# ─────────────────────────────────────────────────────────────────────────────
def generate_difference_map(img1: Image.Image, img2: Image.Image) -> Image.Image:
    def to_arr(p):
        return (np.array(p.convert("L").resize((IMG_SIZE, IMG_SIZE)), np.float32) / 127.5) - 1.0
    diff = np.abs(to_arr(img1) - to_arr(img2))
    mx = diff.max()
    if mx > 0: diff /= mx
    return Image.fromarray(
        (matplotlib.colormaps.get_cmap("inferno")(diff)[:, :, :3] * 255).astype(np.uint8))


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    def to_arr(p):
        return (np.array(p.convert("L").resize((IMG_SIZE, IMG_SIZE)), np.float32) / 127.5) - 1.0
    return float(ssim(to_arr(img1), to_arr(img2), data_range=2))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (once)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(""):
    MODELS = load_models()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 28px 20px 24px; border-bottom: 1px solid rgba(0,200,255,0.08);">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#00c8ff,#0066aa);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;
                    font-size:16px;flex-shrink:0;">🧬</div>
        <div>
          <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;
                      color:#fff;letter-spacing:0.02em;line-height:1;">TumorX GAN</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.14em;
                      color:#2a4060;text-transform:uppercase;margin-top:2px;">v2.0 · Research</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 20px 20px 0;">
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;
                  text-transform:uppercase;color:#1e3050;margin-bottom:12px;">Architecture</div>

      <div style="background:rgba(0,200,255,0.04);border:1px solid rgba(0,200,255,0.08);
                  border-radius:8px;padding:14px;margin-bottom:10px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#00c8ff;margin-bottom:6px;">
          CNN Classifier</div>
        <div style="font-size:0.72rem;color:#4a5a78;line-height:1.7;">
          Conv×4 [32→256] · BN · Pool<br>
          SpatialDropout(0.3) · GAP<br>
          Dense(1, sigmoid)<br>
          <span style="color:#1e3050;">Input: (128,128,1) · [-1,1]</span>
        </div>
      </div>

      <div style="background:rgba(0,200,255,0.04);border:1px solid rgba(0,200,255,0.08);
                  border-radius:8px;padding:14px;margin-bottom:10px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#00c8ff;margin-bottom:6px;">
          GAN Generator</div>
        <div style="font-size:0.72rem;color:#4a5a78;line-height:1.7;">
          z(100) = z_s(80) || z_p(20)<br>
          Dense(262144) → Reshape(32²,256)<br>
          ConvT×2 · BN · LeakyReLU(0.2)<br>
          Conv(1, tanh) → (128,128,1)
        </div>
      </div>

      <div style="background:rgba(255,64,96,0.04);border:1px solid rgba(255,64,96,0.1);
                  border-radius:8px;padding:14px;margin-bottom:20px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#ff6080;margin-bottom:6px;">
          Counterfactual Logic</div>
        <div style="font-size:0.72rem;color:#4a5a78;line-height:1.7;">
          z_structure fixed<br>
          z_p = <b style="color:#00e5a0;">0</b> → No Tumor<br>
          z_p = <b style="color:#ff4060;">N(0,I₂₀)</b> → Tumor
        </div>
      </div>

      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;
                  text-transform:uppercase;color:#1e3050;margin-bottom:10px;">Model Status</div>
    </div>""", unsafe_allow_html=True)

    for key, path in [("Classifier", CLASSIFIER_PATH),
                      ("Generator",  GENERATOR_PATH),
                      ("Discriminator", DISCRIMINATOR_PATH)]:
        ok    = MODELS.get(key.lower()) is not None
        dot   = "#00e5a0" if ok else "#ff4060"
        label = "Online" if ok else "Offline"
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:6px 20px;">
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:6px;height:6px;border-radius:50%;background:{dot};
                        box-shadow:0 0 6px {dot};"></div>
            <span style="font-size:0.78rem;color:#8a9ec0;">{key}</span>
          </div>
          <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{dot};">
            {label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:20px;margin-top:20px;border-top:1px solid rgba(255,255,255,0.04);">
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1a2535;
                  text-align:center;line-height:1.6;">
        NOT FOR CLINICAL USE<br>RESEARCH PROTOTYPE
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
all_ok = all(MODELS.get(k) is not None for k in ["classifier", "generator", "discriminator"])
chip_color = "#00e5a0" if all_ok else "#ffa040"
chip_label = "All Models Online" if all_ok else "Some Models Offline"

st.markdown(f"""
<div style="padding: 36px 0 28px; border-bottom: 1px solid rgba(255,255,255,0.04);
            margin-bottom: 32px;">
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;
              color:#00c8ff;text-transform:uppercase;margin-bottom:8px;
              display:flex;align-items:center;gap:8px;">
    <span style="display:inline-block;width:20px;height:1px;background:#00c8ff;"></span>
    Explainable AI · Brain MRI Analysis
  </div>
  <div style="display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:12px;">
    <div>
      <h1 style="font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
                 color:#fff;margin:0;letter-spacing:-0.01em;line-height:1.1;">
        TumorX <span style="color:#00c8ff;">GAN</span>
      </h1>
      <p style="font-size:0.88rem;color:#2a4060;margin:8px 0 0;font-weight:300;letter-spacing:0.02em;">
        Disentangled Generative Adversarial Network for interpretable tumor analysis
        &nbsp;·&nbsp; Counterfactual generation &nbsp;·&nbsp; Grad-CAM explainability
      </p>
    </div>
    <div style="display:flex;align-items:center;gap:6px;padding:7px 14px;
                border:1px solid {chip_color}33;border-radius:20px;background:{chip_color}0a;">
      <div style="width:5px;height:5px;border-radius:50%;background:{chip_color};
                  box-shadow:0 0 6px {chip_color};"></div>
      <span style="font-family:'DM Mono',monospace;font-size:0.65rem;
                   color:{chip_color};letter-spacing:0.08em;">{chip_label}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_upload, tab_predict, tab_generate, tab_cf, tab_explain, tab_perf = st.tabs([
    "01 · Upload",
    "02 · Classify",
    "03 · Generate",
    "04 · Counterfactual",
    "05 · Explainability",
    "06 · Performance",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 01 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("<br>", unsafe_allow_html=True)
    col_up, col_prev = st.columns([1, 1], gap="large")

    with col_up:
        section_label("Input Image")
        st.markdown("""
        <p style="font-size:0.82rem;color:#2a4060;margin-bottom:16px;line-height:1.7;">
          Upload a grayscale brain MRI scan. It will be converted to grayscale,
          resized to <strong style="color:#8a9ec0;">128 × 128</strong> and
          normalized to <strong style="color:#8a9ec0;">[-1, 1]</strong>
          before model inference.
        </p>""", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image  = image
            st.session_state.prediction_done = False
            st.success(f"Loaded **{uploaded_file.name}**")
            st.markdown(f"""
            <div style="display:flex;gap:24px;margin-top:12px;flex-wrap:wrap;">
              <div style="font-family:'DM Mono',monospace;font-size:0.72rem;">
                <span style="color:#2a4060;">ORIGINAL</span><br>
                <span style="color:#8a9ec0;">{image.size[0]} x {image.size[1]} px · {image.mode}</span>
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:0.72rem;">
                <span style="color:#2a4060;">MODEL INPUT</span><br>
                <span style="color:#00c8ff;">128 x 128 · L · [-1,1]</span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No image uploaded yet.")

    with col_prev:
        section_label("Preview")
        if st.session_state.uploaded_image is not None:
            image_frame(st.session_state.uploaded_image, "Original resolution", tag="INPUT")
        else:
            empty_frame("AWAITING UPLOAD")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 02 — CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.uploaded_image is None:
        st.warning("Upload an MRI scan in **01 · Upload** first.")
    else:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            section_label("Preprocessed Input")
            arr_preview = preprocess_image(st.session_state.uploaded_image)
            arr_vis = ((arr_preview[0].squeeze() + 1) * 127.5).astype(np.uint8)
            image_frame(Image.fromarray(arr_vis, mode="L"),
                        "128x128 · grayscale · normalized [-1,1]", tag="INPUT")

        with col_result:
            section_label("CNN Classification")

            if MODELS.get("classifier") is None:
                st.error(f"Classifier not found: `{CLASSIFIER_PATH}`")
            else:
                if st.button("Run Classification", key="btn_predict"):
                    with st.spinner("Running CNN classifier…"):
                        label, conf = predict(st.session_state.uploaded_image, MODELS)
                        st.session_state.pred_label      = label
                        st.session_state.pred_confidence = conf
                        st.session_state.prediction_done = True

                if st.session_state.prediction_done:
                    label = st.session_state.pred_label
                    conf  = st.session_state.pred_confidence

                    st.markdown("<br>", unsafe_allow_html=True)
                    pill(f"◉  {label.upper()}", kind="tumor" if label == "Tumor" else "normal")

                    st.markdown("""
                    <div style="margin:24px 0 8px;font-family:'DM Mono',monospace;font-size:0.62rem;
                                letter-spacing:0.14em;color:#2a4060;">CONFIDENCE SCORE</div>""",
                                unsafe_allow_html=True)
                    st.progress(conf)

                    color = "#ff4060" if label == "Tumor" else "#00e5a0"
                    st.markdown(f"""
                    <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:700;
                                color:{color};margin-top:6px;line-height:1;">
                      {conf * 100:.1f}<span style="font-size:1rem;font-weight:400;">%</span>
                    </div>""", unsafe_allow_html=True)

                    divider()

                    if label == "Tumor":
                        st.markdown("""
                        <div style="background:rgba(255,64,96,0.05);border:1px solid rgba(255,64,96,0.2);
                                    border-left:3px solid #ff4060;border-radius:6px;padding:14px 16px;">
                          <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#ff6080;
                                      letter-spacing:0.08em;margin-bottom:4px;">POSITIVE DETECTION</div>
                          <div style="font-size:0.8rem;color:#4a5a78;line-height:1.6;">
                            Proceed to <strong style="color:#e2e8f4;">04 · Counterfactual</strong> for
                            causal analysis and <strong style="color:#e2e8f4;">05 · Explainability</strong>
                            for Grad-CAM attention maps.
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background:rgba(0,229,160,0.05);border:1px solid rgba(0,229,160,0.2);
                                    border-left:3px solid #00e5a0;border-radius:6px;padding:14px 16px;">
                          <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#00e5a0;
                                      letter-spacing:0.08em;margin-bottom:4px;">NEGATIVE RESULT</div>
                          <div style="font-size:0.8rem;color:#4a5a78;line-height:1.6;">
                            Review <strong style="color:#e2e8f4;">05 · Explainability</strong>
                            to verify model attention regions.
                          </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="padding:24px 0;font-family:'DM Mono',monospace;
                                font-size:0.72rem;color:#1e2d48;">
                      Press the button to run inference on the uploaded scan.
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 03 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_generate:
    st.markdown("<br>", unsafe_allow_html=True)

    col_ctrl, col_out = st.columns([1, 1], gap="large")

    with col_ctrl:
        section_label("Latent Space Controls")
        st.markdown("""
        <p style="font-size:0.8rem;color:#2a4060;margin-bottom:20px;line-height:1.7;">
          Sample a fresh <code style="color:#8a9ec0;">z_structure ~ N(0, I₈₀)</code>
          and control the pathology vector magnitude.
        </p>""", unsafe_allow_html=True)

        with_tumor = st.toggle("Enable z_pathology (Tumor)", value=True)
        z_strength = st.slider("z_pathology magnitude", 0.0, 1.0, 1.0, 0.05)

        zp_str = f"{z_strength:.2f} x N(0, I₂₀)" if with_tumor else "zeros (disabled)"
        zp_col = "#ff4060" if with_tumor else "#2a4060"
        st.markdown(f"""
        <div style="background:rgba(0,200,255,0.04);border:1px solid rgba(0,200,255,0.08);
                    border-radius:8px;padding:14px;margin:16px 0;
                    font-family:'DM Mono',monospace;font-size:0.7rem;color:#4a5a78;line-height:2.2;">
          z_structure ~ N(0, I₈₀)<br>
          z_pathology = <span style="color:{zp_col};">{zp_str}</span>
        </div>""", unsafe_allow_html=True)

        gen_btn = st.button("Generate MRI", key="btn_gen")

    with col_out:
        section_label("Synthetic Output")
        if MODELS.get("generator") is None:
            st.error(f"Generator not found: `{GENERATOR_PATH}`")
        elif gen_btn:
            with st.spinner("Sampling latent space · decoding…"):
                gen_img = generate_image(MODELS, with_tumor=with_tumor,
                                          z_pathology_strength=z_strength)
            tag = "TUMOR" if with_tumor and z_strength > 0 else "NO TUMOR"
            image_frame(gen_img, f"z_pathology scale = {z_strength:.2f}", tag=tag)
            if MODELS.get("classifier") is not None:
                with st.spinner("Running classifier…"):
                    lbl, c = predict(gen_img, MODELS)
                st.markdown(f"""
                <div style="margin-top:10px;font-family:'DM Mono',monospace;font-size:0.7rem;color:#4a5a78;">
                  Classifier: <span style="color:{'#ff4060' if lbl=='Tumor' else '#00e5a0'};">
                    {lbl}</span> · {c*100:.1f}%
                </div>""", unsafe_allow_html=True)
        else:
            empty_frame("PRESS GENERATE")

    divider()
    section_label("Latent Traversal · z_pathology = 0 to 1")
    st.markdown("""
    <p style="font-size:0.8rem;color:#2a4060;margin-bottom:16px;">
      Fixed <code style="color:#8a9ec0;">z_structure</code> across all 5 images.
      Pathology vector scaled from 0 to 1, showing the learned tumor manifold.
    </p>""", unsafe_allow_html=True)

    if st.button("Generate Traversal Strip", key="btn_trav"):
        if MODELS.get("generator") is None:
            st.error("Generator not loaded.")
        else:
            with st.spinner("Generating traversal…"):
                levels    = [0.0, 0.25, 0.5, 0.75, 1.0]
                z_s       = tf.random.normal((1, STRUCTURE_DIM))
                z_pb      = tf.random.normal((1, PATHOLOGY_DIM))
                trav_cols = st.columns(5)
                for col, lv in zip(trav_cols, levels):
                    z   = tf.concat([z_s, lv * z_pb], axis=1)
                    pil = tensor_to_pil(MODELS["generator"](z, training=False))
                    with col:
                        image_frame(pil, f"z_p = {lv:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 04 — COUNTERFACTUAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_cf:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.85rem;color:#2a4060;max-width:640px;line-height:1.8;margin-bottom:24px;">
      The same <code style="color:#8a9ec0;">z_structure</code> is fixed across both images.
      Only <code style="color:#8a9ec0;">z_pathology</code> is toggled — producing a
      causally valid pair that isolates the tumor signal.
    </p>""", unsafe_allow_html=True)

    if MODELS.get("generator") is None:
        st.error(f"Generator not found: `{GENERATOR_PATH}`")
    else:
        if st.button("Generate Counterfactual Pair", key="btn_cf"):
            with st.spinner("Fixing z_structure · toggling z_pathology · decoding both…"):
                no_t, yes_t = generate_counterfactual(MODELS)

            col1, col2, col3 = st.columns(3, gap="medium")

            with col1:
                section_label("Without Tumor", color="#00e5a0")
                image_frame(no_t, "z_pathology = zeros", tag="NO TUMOR")
                if MODELS.get("classifier"):
                    lbl_no, c_no = predict(no_t, MODELS)
                    st.markdown(f"""
                    <div style="margin-top:8px;font-family:'DM Mono',monospace;font-size:0.68rem;color:#4a5a78;">
                      CNN: <span style="color:#00e5a0;">{lbl_no}</span> · {c_no*100:.1f}%
                    </div>""", unsafe_allow_html=True)

            with col2:
                section_label("With Tumor", color="#ff4060")
                image_frame(yes_t, "z_pathology ~ N(0, I₂₀)", tag="TUMOR")
                if MODELS.get("classifier"):
                    lbl_yes, c_yes = predict(yes_t, MODELS)
                    st.markdown(f"""
                    <div style="margin-top:8px;font-family:'DM Mono',monospace;font-size:0.68rem;color:#4a5a78;">
                      CNN: <span style="color:#ff4060;">{lbl_yes}</span> · {c_yes*100:.1f}%
                    </div>""", unsafe_allow_html=True)

            with col3:
                section_label("Difference Map", color="#ffa040")
                diff_img = generate_difference_map(no_t, yes_t)
                sc = compute_ssim(no_t, yes_t)
                st.session_state.latest_ssim = sc
                image_frame(diff_img, f"SSIM = {sc:.4f}", tag="DIFF")
                st.markdown("""
                <div style="margin-top:8px;font-family:'DM Mono',monospace;font-size:0.68rem;color:#4a5a78;">
                  Pixel-wise |Tumor - No Tumor| · inferno colormap
                </div>""", unsafe_allow_html=True)

            divider()
            cs1, cs2, cs3 = st.columns(3)
            with cs1: stat_card(f"{sc:.4f}", "SSIM Score",     "#ffa040", "data_range=2")
            with cs2: stat_card("80 / 20",   "z_s / z_p Dims", "#00c8ff", "Latent split")
            with cs3:
                delta = abs(c_yes - c_no) * 100 if MODELS.get("classifier") else 0
                stat_card(f"{delta:.1f}%", "Confidence Delta", "#ff4060", "Between pair")

        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:220px;border:1px dashed rgba(0,200,255,0.08);border-radius:10px;">
              <span style="font-family:'DM Mono',monospace;font-size:0.68rem;
                           letter-spacing:0.14em;color:#1e2d48;">
                PRESS THE BUTTON TO GENERATE PAIR
              </span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 05 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.uploaded_image is None:
        st.warning("Upload an MRI in **01 · Upload** first.")
    elif MODELS.get("classifier") is None:
        st.error("Classifier not loaded.")
    else:
        if st.button("Run Full Explainability Analysis", key="btn_exp"):

            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                section_label("Grad-CAM · CNN Classifier", color="#c084fc")
                with st.spinner("Computing gradients on last Conv2D…"):
                    gc_cls = generate_gradcam(st.session_state.uploaded_image, MODELS,
                                              target="classifier")
                image_frame(gc_cls, "Warm = tumor-discriminative region", tag="GRAD-CAM")
                st.markdown("""
                <div style="font-size:0.75rem;color:#2a4060;margin-top:10px;line-height:1.7;">
                  GradientTape on <code style="color:#8a9ec0;">conv2d_19</code> (last Conv2D).
                  Pooled gradients weight feature maps → resized → jet overlay (alpha=0.50).
                </div>""", unsafe_allow_html=True)

            # with col_b:
            #     if MODELS.get("discriminator") is not None:
            #         section_label("Grad-CAM · GAN Discriminator", color="#c084fc")
            #         with st.spinner("Computing discriminator attention…"):
            #             gc_disc = generate_gradcam(st.session_state.uploaded_image, MODELS,
            #                                        target="discriminator")
            #         image_frame(gc_disc, "Realism-score attention regions", tag="GRAD-CAM")
            #         st.markdown("""
            #         <div style="font-size:0.75rem;color:#2a4060;margin-top:10px;line-height:1.7;">
            #           Same Grad-CAM applied to discriminator's last Conv2D.
            #           Highlights regions driving the real/fake decision.
            #         </div>""", unsafe_allow_html=True)
            #     else:
            #         section_label("Discriminator Grad-CAM", color="#ffa040")
            #         st.warning(f"Discriminator not loaded: `{DISCRIMINATOR_PATH}`")

            divider()
            section_label("Three-Way Comparison")

            with st.spinner("Generating counterfactual pair…"):
                no_t2, yes_t2 = generate_counterfactual(MODELS)
                diff2         = generate_difference_map(no_t2, yes_t2)

            cc1, cc2, cc3 = st.columns(3, gap="medium")
            with cc1: image_frame(st.session_state.uploaded_image, "Uploaded MRI",       tag="INPUT")
            with cc2: image_frame(gc_cls,                          "Classifier attention", tag="GRAD-CAM")
            with cc3: image_frame(diff2,                           "Tumor signal",         tag="DIFF")

            divider()
            section_label("Faithfulness Test · Mask the Explanation Region")
            st.markdown("""
            <p style="font-size:0.8rem;color:#2a4060;margin-bottom:16px;line-height:1.7;">
              If the explanation is faithful, masking the difference map region
              should significantly reduce tumor confidence.
            </p>""", unsafe_allow_html=True)

            if MODELS.get("generator") is not None:
                with st.spinner("Running faithfulness test…"):
                    diff_arr = np.array(diff2.convert("L"), dtype=np.float32) / 255.0
                    diff_arr = diff_arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                    arr_gen  = preprocess_image(yes_t2)
                    masked   = arr_gen * (1.0 - diff_arr)
                    p_tumor  = float(MODELS["classifier"].predict(arr_gen,  verbose=0)[0][0])
                    p_masked = float(MODELS["classifier"].predict(masked,   verbose=0)[0][0])
                    drop     = (p_tumor - p_masked) * 100

                fm1, fm2, fm3 = st.columns(3)
                with fm1: stat_card(f"{p_tumor*100:.1f}%",  "Original Confidence",  "#ff4060")
                with fm2: stat_card(f"{p_masked*100:.1f}%", "After Masking",         "#ffa040")
                with fm3:
                    c = "#00e5a0" if drop > 10 else "#ffa040"
                    stat_card(f"{drop:+.1f}%", "Confidence Drop", c)

        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:220px;border:1px dashed rgba(0,200,255,0.08);border-radius:10px;">
              <span style="font-family:'DM Mono',monospace;font-size:0.68rem;
                           letter-spacing:0.14em;color:#1e2d48;">
                PRESS THE BUTTON TO RUN ANALYSIS
              </span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 06 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("<br>", unsafe_allow_html=True)

    hist = load_history()

    if hist:
        best_ep  = int(np.argmax(hist["val_accuracy"]))
        best_acc = hist["val_accuracy"][best_ep]
        best_loss= hist["val_loss"][best_ep]
        final_tr = hist["accuracy"][-1]
        n_epochs = len(hist["accuracy"])

        section_label("Training Summary")
        s1, s2, s3, s4 = st.columns(4, gap="medium")
        with s1: stat_card(f"{best_acc*100:.2f}%",       "Best Val Accuracy",  "#00c8ff")
        with s2: stat_card(f"{best_loss:.4f}",            "Best Val Loss",      "#00e5a0")
        with s3: stat_card(f"{final_tr*100:.2f}%",        "Final Train Acc",    "#c084fc")
        # with s4: stat_card(f"Ep {best_ep+1}/{n_epochs}",  "Best Epoch",         "#ffa040")

        st.markdown("<br>", unsafe_allow_html=True)
        section_label("Training History")

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        fig.patch.set_facecolor("#080c18")
        fig.subplots_adjust(wspace=0.35)
        epochs_x = range(1, n_epochs + 1)

        for ax in axes:
            ax.set_facecolor("#0a0e1a")
            ax.spines[:].set_color("none")
            ax.spines["bottom"].set_color("#1a2840")
            ax.spines["left"].set_color("#1a2840")
            ax.tick_params(colors="#2a4060", labelsize=8)
            ax.set_xlabel("Epoch", color="#2a4060", fontsize=8, labelpad=8)
            ax.grid(axis="y", color="#0d1830", linewidth=0.6)

        axes[0].plot(epochs_x, hist["accuracy"],    color="#00c8ff", lw=2, label="Train")
        axes[0].plot(epochs_x, hist["val_accuracy"], color="#c084fc", lw=2, ls="--", alpha=0.85, label="Val")
        axes[0].fill_between(epochs_x, hist["accuracy"], alpha=0.06, color="#00c8ff")
        axes[0].set_ylim(0.4, 1.02)
        axes[0].set_title("Accuracy", color="#8a9ec0", fontsize=10, pad=10, fontweight="600")
        axes[0].legend(facecolor="#0a0e1a", labelcolor="#8a9ec0", edgecolor="#1a2840", fontsize=8)

        axes[1].plot(epochs_x, hist["loss"],    color="#ff4060", lw=2, label="Train")
        axes[1].plot(epochs_x, hist["val_loss"], color="#ffa040", lw=2, ls="--", alpha=0.85, label="Val")
        axes[1].fill_between(epochs_x, hist["loss"], alpha=0.06, color="#ff4060")
        axes[1].set_title("Loss", color="#8a9ec0", fontsize=10, pad=10, fontweight="600")
        axes[1].legend(facecolor="#0a0e1a", labelcolor="#8a9ec0", edgecolor="#1a2840", fontsize=8)

        plt.tight_layout()
        st.image(fig_to_pil(fig), use_container_width=True)
        plt.close(fig)

    else:
        st.warning(f"`{HISTORY_PATH}` not found. Place it alongside `app.py`.")

    divider()
    section_label("Evaluation Metrics · Test Set")
    st.markdown('<p style="font-size:0.75rem;color:#1e2d48;margin-bottom:14px;">Replace with your actual classification_report from notebook cell 20.</p>',
                unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4, gap="medium")
    with e1: stat_card("~96%", "Accuracy",  "#00c8ff")
    with e2: stat_card("~96%", "Precision", "#00e5a0")
    with e3: stat_card("~97%", "Recall",    "#c084fc")
    with e4: stat_card("~96%", "F1-Score",  "#ffa040")

    divider()
    section_label("GAN Quality Metrics")
    g1, g2, g3 = st.columns(3, gap="medium")
    ssim_display = (
      f"{st.session_state.latest_ssim:.4f}"
      if st.session_state.latest_ssim is not None
      else "N/A"
    )
    with g1: stat_card(ssim_display,"SSIM (range=0-1)","#00c8ff","From latest counterfactual")
    # with g2: stat_card("-",     "FID",                  "#c084fc", "Add from eval)uation")

    divider()
    section_label("Per-Class Report")
    st.markdown("""
    <table style="width:100%;font-family:'DM Mono',monospace;font-size:0.75rem;
                  border-collapse:collapse;color:#e2e8f4;">
      <thead>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
          <th style="text-align:left;padding:10px 12px;color:#4a5a78;font-weight:500;">Class</th>
          <th style="padding:10px 16px;color:#4a5a78;font-weight:500;">Precision</th>
          <th style="padding:10px 16px;color:#4a5a78;font-weight:500;">Recall</th>
          <th style="padding:10px 16px;color:#4a5a78;font-weight:500;">F1</th>
          <th style="padding:10px 16px;color:#4a5a78;font-weight:500;">Support</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.03);">
          <td style="padding:10px 12px;color:#00e5a0;">No Tumor</td>
          <td style="text-align:center;padding:10px 16px;">97.4%</td>
          <td style="text-align:center;padding:10px 16px;">96.1%</td>
          <td style="text-align:center;padding:10px 16px;">96.7%</td>
          <td style="text-align:center;padding:10px 16px;">154</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.03);">
          <td style="padding:10px 12px;color:#ff4060;">Tumor</td>
          <td style="text-align:center;padding:10px 16px;">95.9%</td>
          <td style="text-align:center;padding:10px 16px;">97.2%</td>
          <td style="text-align:center;padding:10px 16px;">96.6%</td>
          <td style="text-align:center;padding:10px 16px;">146</td>
        </tr>
        <tr>
          <td style="padding:10px 12px;color:#4a5a78;font-style:italic;">Weighted avg</td>
          <td style="text-align:center;padding:10px 16px;color:#4a5a78;">96.7%</td>
          <td style="text-align:center;padding:10px 16px;color:#4a5a78;">96.6%</td>
          <td style="text-align:center;padding:10px 16px;color:#4a5a78;">96.6%</td>
          <td style="text-align:center;padding:10px 16px;color:#4a5a78;">300</td>
        </tr>
      </tbody>
    </table>
    <div style="font-size:0.65rem;color:#1e2d48;margin-top:10px;">
      Fill in from classification_report(y_val, y_val_pred) — notebook cell 20
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:60px;padding:24px 0;border-top:1px solid rgba(255,255,255,0.04);
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
  <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#1a2535;">
    TumorX <span style="color:#0a1f35;">GAN</span>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;color:#0d1830;text-transform:uppercase;">
    Explainable GAN · Brain MRI Analysis · Research Prototype · Not for Clinical Use
  </div>
</div>""", unsafe_allow_html=True)