import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
import matplotlib.cm as cm

# ==============================
# LOAD MODELS
# ==============================
generator = load_model("gan_checkpoints_V2_h5/generator_epoch_60.h5")
classifier = load_model("tumor_classifier_axial.h5")
classifier.trainable = False

# ==============================
# CONSTANTS
# ==============================
STRUCTURE_DIM = 80
PATHOLOGY_DIM = 20

# ==============================
# UTILS
# ==============================

def normalize(img):
    return ((img + 1) / 2.0)[0].numpy().squeeze()

def predict(img):
    return classifier(img).numpy()[0][0]

def apply_hot_colormap(diff):
    diff = diff[0].numpy().squeeze()
    diff = diff / (diff.max() + 1e-8)
    colored = cm.hot(diff)
    return colored

# ------------------------------
# Grad-CAM
# ------------------------------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

def grad_cam(model, img, layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    # Resize (IMPORTANT)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (128, 128)
    )

    heatmap = tf.squeeze(heatmap)

    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    # Convert image to [0,1]
    img = normalize(img)

    # Convert grayscale → RGB
    img = np.stack([img]*3, axis=-1)

    # Heatmap → 0-255
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap (JET like your backend)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert BGR → RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Normalize
    heatmap = heatmap / 255.0

    # Overlay
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay

def focus_map(img, heatmap, threshold=0.6):
    mask = heatmap > threshold

    img_np = normalize(img)
    img_np = np.stack([img_np]*3, axis=-1)

    focused = img_np * mask[..., np.newaxis]
    return focused

# ------------------------------
# GENERATION
# ------------------------------
def generate(label=1):
    z_structure = tf.random.normal((1, STRUCTURE_DIM))
    z_pathology = tf.random.normal((1, PATHOLOGY_DIM)) * label
    z = tf.concat([z_structure, z_pathology], axis=1)
    return generator(z, training=False)

def counterfactual():
    z_structure = tf.random.normal((1, STRUCTURE_DIM))

    z_no = tf.concat([z_structure, tf.zeros((1, PATHOLOGY_DIM))], axis=1)
    z_yes = tf.concat([z_structure, tf.random.normal((1, PATHOLOGY_DIM))], axis=1)

    return generator(z_no), generator(z_yes)

def counterfactual_sequence(steps=10):
    z_structure = tf.random.normal((1, STRUCTURE_DIM))
    z_pathology = tf.random.normal((1, PATHOLOGY_DIM))

    imgs = []

    for i in range(steps):
        level = i / (steps - 1)
        z = tf.concat([z_structure, level * z_pathology], axis=1)
        img = generator(z, training=False)
        imgs.append(img)

    return imgs

def diff_map(a, b):
    d = tf.abs(b - a)
    d /= tf.reduce_max(d) + 1e-8
    return d

# ==============================
# UI SETUP
# ==============================
st.set_page_config(layout="wide")
st.title("🧠 Explainable GAN for Brain MRI")

tabs = st.tabs([
    "MRI Generation",
    "Counterfactual",
    "Latent Traversal",
    "Explainability"
])

# ==============================
# 1. MRI GENERATION
# ==============================
with tabs[0]:
    st.subheader("MRI Generation")

    col1, col2 = st.columns([1,2])

    with col1:
        label = st.radio("Type", ["No Tumor", "Tumor"])
        show_cam = st.checkbox("Show Grad-CAM")
        alpha = st.slider("Grad-CAM Intensity", 0.0, 1.0, 0.4)

        if st.button("Generate"):
            lbl = 1 if label == "Tumor" else 0
            st.session_state.img = generate(lbl)

    with col2:
        if "img" in st.session_state:
            img = st.session_state.img
            st.image(normalize(img), use_container_width=True)

            conf = predict(img)
            st.metric("Tumor Confidence", f"{conf:.3f}")

            if show_cam:
                layer_name = get_last_conv_layer(classifier)
                heatmap = grad_cam(classifier, img, layer_name)

                overlay = overlay_gradcam(img, heatmap, alpha)

                st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

# ==============================
# 2. COUNTERFACTUAL
# ==============================
with tabs[1]:
    st.subheader("Counterfactual Explanation")
    animate_cf = st.checkbox("Animate Counterfactual")
    speed = st.slider("Animation Speed", 0.1, 1.0, 0.5)
    steps = st.slider("Steps", 3, 15, 8)

    if st.button("Generate Counterfactual"):
        st.session_state.no, st.session_state.yes = counterfactual()

    if "no" in st.session_state:
        img_no = st.session_state.no
        img_yes = st.session_state.yes
        diff = diff_map(img_no, img_yes)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(normalize(img_no))
            st.write(f"Classifier No Tumor: {predict(img_no):.8f}")
        
        with c2:
            st.image(normalize(img_yes))
            st.write(f"Classifier Tumor: {predict(img_yes):.8f}")

        with c3:
            st.image(
                apply_hot_colormap(diff),
                use_container_width=True
            )
            st.caption("Difference Map")
            

    
    if animate_cf:

        st.markdown("### 🎥 Counterfactual Transition")

        imgs = counterfactual_sequence(steps)

        frame = st.empty()
        text = st.empty()

        for img in imgs:
            frame.image(normalize(img), use_container_width=True)

            conf = predict(img)
            st.caption(f"Tumor Confidence: {conf:.4f}")

            time.sleep(speed)

# ==============================
# 3. LATENT TRAVERSAL (ANIMATION)
# ==============================
with tabs[2]:
    st.subheader("Latent Traversal (Pathology Evolution)")

    steps = st.slider("Steps", 3, 10, 5)
    animate = st.checkbox("Animate")
    show_cam = st.checkbox("Show Grad-CAM Animation")

    if st.button("Generate Sequence"):

        z_structure = tf.random.normal((1, STRUCTURE_DIM))
        z_path = tf.random.normal((1, PATHOLOGY_DIM))

        imgs = []

        for i in range(steps):
            level = i / (steps - 1)
            z = tf.concat([z_structure, level * z_path], axis=1)
            img = generator(z, training=False)
            imgs.append(img)

        st.session_state.sequence = imgs

    if "sequence" in st.session_state:

        imgs = st.session_state.sequence

        if animate:
            frame = st.empty()
            layer_name = get_last_conv_layer(classifier)
            for img in imgs:
                if show_cam:
                    heatmap = grad_cam(classifier, img, layer_name)
                    overlay = overlay_gradcam(img, heatmap, 0.5)
                    frame.image(overlay)
                else:
                    frame.image(normalize(img))

                time.sleep(0.5)
        else:
            cols = st.columns(len(imgs))
            for i, img in enumerate(imgs):
                cols[i].image(normalize(img))
                cols[i].caption(f"{i/(len(imgs)-1):.2f}")
                
        # ==============================
        # 📊 Confidence Graph (Step 7)
        # ==============================
        st.markdown("### 📈 Tumor Confidence vs Pathology Level")

        import matplotlib.pyplot as plt

        confidences = [predict(i) for i in imgs]

        fig, ax = plt.subplots()
        ax.plot(confidences, marker='o')
        ax.set_xlabel("Traversal Step")
        ax.set_ylabel("Tumor Confidence")
        ax.set_title("Confidence Evolution")

        st.pyplot(fig)


# ==============================
# 4. EXPLAINABILITY PANEL
# ==============================
with tabs[3]:
    st.subheader("Explainability Panel")

    if st.button("Run Full Explanation"):
        img_no, img_yes = counterfactual()

        diff = diff_map(img_no, img_yes)

        masked = img_yes * (1 - diff)

        p_before = predict(img_yes)
        p_after = predict(masked)

        st.session_state.exp = (img_yes, diff, masked, p_before, p_after)

    if "exp" in st.session_state:
        img, diff, masked, p1, p2 = st.session_state.exp

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(normalize(img))
            st.caption("Original")

        with c2:
            st.image(normalize(diff))
            st.caption("Important Regions")

        with c3:
            st.image(normalize(masked))
            st.caption("Masked")
        
        st.markdown("### 🔬 Attention vs True Change")

        layer_name = get_last_conv_layer(classifier)
        heatmap = grad_cam(classifier, img, layer_name)

        overlay = overlay_gradcam(img, heatmap, 0.5)

        c1, c2 = st.columns(2)

        with c1:
            st.image(overlay)
            st.caption("Grad-CAM (Model Attention)")

        with c2:
            st.image(normalize(diff))
            st.caption("Difference Map (True Change)")
    
        st.markdown("### 📊 Explanation")

        st.metric("Confidence Before", f"{p1:.3f}")
        st.metric("Confidence After Masking", f"{p2:.3f}")

        st.success(
            "Large drop → model relies on highlighted regions"
        )
        # ==============================
        # 🔍 Focus Map (Step 6)
        # ==============================
        st.markdown("### 🎯 Focused Important Regions")

        threshold = st.slider("Focus Threshold", 0.0, 1.0, 0.6, key="focus_slider")

        focused = focus_map(img, heatmap, threshold)

        st.image(focused, caption="Focused Regions Only")