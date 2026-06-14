# Explainable GAN-based Brain MRI Tumor Project

This repository contains code for training a CNN classifier and a customized GAN
on a dataset of brain MRI scans, some containing tumors and some without. The
goal is to generate realistic MRI images with controllable tumor presence, and
to analyze GAN behavior using counterfactuals and explainability methods such as
Grad-CAM.

The core focus of this project is **explainable generative adversarial networks**. 
By splitting the latent vector into separate "structure" and "pathology"
dimensions, and integrating a frozen tumor classifier into the generator loss,
we force the GAN to encode tumor-specific features in a controllable way. This
allows us to generate counterfactual image pairs (same anatomy with/without
tumor) and to inspect which pixels the generator alters when toggling the
pathology component. Additionally, Grad-CAM is applied to both the classifier
and discriminator to visualize where each model attends when making decisions.

The novelty lies in combining a disentangled latent space with classifier-
guided training to produce interpretable tumor manipulations, offering a more
transparent understanding of what the GAN has learned compared to a standard
black-box model.

## 🚀 Key Features

- 🧠 **Disentangled Latent Space**
  - Separates anatomical structure and pathology using structured latent vectors.
  - Inspired by explainable GAN frameworks like medXGAN :contentReference[oaicite:0]{index=0}.

- 🔍 **Classifier-Guided GAN Training**
  - Frozen CNN classifier integrated into generator loss.
  - Forces medically meaningful tumor representations.

- 🔁 **Counterfactual MRI Generation**
  - Generate same brain with/without tumor.
  - Enables causal-style reasoning instead of static saliency maps.

- 🎯 **Explainability Modules**
  - Grad-CAM on classifier and discriminator.
  - Pixel-wise difference maps between counterfactuals.
  - Latent traversal visualization.

- 📊 **Evaluation Metrics**
  - Classification: Accuracy, Precision, Recall, F1-score
  - GAN Quality: SSIM (Structural Similarity)
  - Explainability: Localization behavior via difference maps

- 🖥️ **Interactive UI Dashboard**
  - Multi-tab interface for visual exploration of results
  - Real-time switching between:
    - Original images
    - Generated images
    - Counterfactual pairs
    - Grad-CAM heatmaps

## 🧠 Method Overview

The system follows a hybrid pipeline combining discriminative and generative modeling:

1. Train a CNN classifier for tumor detection.
2. Freeze the classifier to act as a supervisory signal.
3. Train a GAN where:
   - Latent vector is split into:
     - `z_structure` → anatomy
     - `z_pathology` → tumor features
4. Generator learns:
   - Realism (via discriminator)
   - Medical correctness (via classifier loss)
5. Generate counterfactuals by:
   - Keeping `z_structure` constant
   - Modifying `z_pathology`

This approach aligns with modern explainable GAN frameworks that leverage latent space manipulation for semantic interpretation :contentReference[oaicite:1]{index=1}.


## 🔍 Explainability Pipeline

The project implements multiple complementary explainability methods:

### 1. Counterfactual Generation
- Generate paired MRI scans:
  - Same anatomy
  - With vs without tumor
- Helps identify causal tumor regions

### 2. Difference Mapping
- Pixel-wise subtraction between counterfactual pairs
- Highlights regions altered by pathology

### 3. Grad-CAM Visualization
- Applied on:
  - CNN classifier → tumor attention
  - GAN discriminator → realism focus

### 4. Latent Space Traversal
- Smooth interpolation of pathology vector
- Observes how tumor gradually appears

This multi-level explainability is stronger than traditional saliency maps, which only localize attention but do not explain feature evolution.

## 🖥️ User Interface (UI)

The project includes an interactive UI built using Streamlit for visual analysis.

### 🔹 Features

-  Upload or select MRI samples
-  Tumor classification results (real-time)
-  GAN-generated MRI visualization
-  Counterfactual comparison:
  - Tumor vs No Tumor (same anatomy)
-  Grad-CAM heatmaps overlay
-  Explainability tab:
  - Difference maps
  - Model attention regions

### 🔹 UI Tabs

1. **Dataset Viewer**
   - Browse tumor / no tumor images

2. **Classifier Results**
   - Prediction + confidence scores

3. **GAN Generator**
   - View synthetic MRI outputs

4. **Counterfactual Explorer**
   - Side-by-side comparison
   - Same brain, different pathology

5. **Explainability Dashboard**
   - Grad-CAM (classifier + discriminator)
   - Difference heatmaps

### 🔹 Design Improvements (your updates)

- Uniform image sizing and alignment
- Responsive layout (no scrolling required)
- Consistent visualization grid across tabs
- Smooth transitions between views

The UI transforms the project from a research prototype into an interpretable system usable for demonstration and analysis.

## 📊 Results Summary

- Classification Accuracy: ~96%
- Strong balance between precision and recall for tumor detection
- GAN successfully generates structurally consistent MRIs
- Counterfactual images preserve anatomy while altering pathology

### Key Insight:
The generator learns **clinically meaningful tumor regions**, not just visual noise — validated through:
- Grad-CAM overlap
- Difference maps
- Classifier consistency on generated samples

## Structure

- `xgan.ipynb` – Main Jupyter notebook with all data loading, preprocessing,
  classifier training, GAN training, evaluation, Grad-CAM visualizations, and
  counterfactual generation.
- `dataset/` – Folder containing two subfolders:
  - `tumor/` – MRI scans labeled as containing tumors.
  - `no_tumor/` – MRI scans without tumors.
- `train_idx.npy`, `val_idx.npy` – Saved indices for deterministic train/validation split.
- `tumor_classifier_axial.h5` – Saved CNN classifier model.
- `classifier_history.json` – Training history for the classifier.
- `gan_checkpoints_V2_h5/` – Saved generator and discriminator H5 checkpoints every 10 epochs.
- `gan_generated_images_V2_h5/` – Generated images from the GAN during training.
- `outputs_gradcam/` – Grad-CAM visualizations for classifier and discriminator.

## 🧩 Novel Contributions

- Explicit disentanglement of anatomy and pathology in GAN latent space
- Integration of frozen classifier for medically guided generation
- Counterfactual MRI synthesis for causal explainability
- Dual explainability:
  - Generative (difference maps)
  - Gradient-based (Grad-CAM)
- End-to-end pipeline with visualization UI

This bridges the gap between:
➡️ Black-box GANs  
➡️ Clinically interpretable AI systems

## Requirements

Create a virtual environment and install dependencies (e.g.):

```bash
python -m venv venv
.\nvenv\Scripts\activate
pip install -r requirements.txt
```

### Example `requirements.txt`
```
tensorflow
numpy
opencv-python
matplotlib
scikit-learn
seaborn
scikit-image
```

## Usage

Open `xgan.ipynb` in Jupyter Notebook or JupyterLab. Run cells sequentially to:

1. Load and preprocess the dataset.
2. Train the CNN classifier and save the model.
3. Visualize classifier performance and Grad-CAM explanations.
4. Define and train the GAN with separate latent structure and pathology
   components, integrating the frozen classifier to guide pathology encoding.
5. Generate and inspect counterfactual image pairs, Grad-CAM on generated
   samples, and calculate similarity metrics.

## ⚠️ Limitations & Future Work

- Limited dataset size may affect GAN diversity
- Reconstruction quality depends on latent space richness
- Currently binary classification (Tumor vs No Tumor)

### Future Improvements
- Multi-class tumor classification
- Higher resolution GANs (128×128 or 256×256)
- Integration of attention-based GANs (WGAN-GP + attention)
- Quantitative explainability metrics (IoU, attribution overlap)
- Clinical validation with expert annotations

## License & Copyright

```
Copyright (c) 2026 Sam T James

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Notes

- Modify latent dimensions and training parameters inside the notebook as
  needed.
- The classifier can be reused for other MRI tasks with minimal changes.

Happy experimenting!
