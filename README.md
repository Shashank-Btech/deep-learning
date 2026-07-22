# 🖼️ CIFAR-10 Image Classifier — Transfer Learning with ResNet-20

> **Goal:** A production-ready deep learning image classifier using PyTorch with a pre-trained ResNet-20, data augmentation, a FastAPI web server, human/text rejection, and custom dataset support.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.4%25-brightgreen)](outputs/training_curves.png)

---

## 📋 Project Overview

This project implements a **CIFAR-10 image classifier** using **Transfer Learning** on a pre-trained **ResNet-20** (trained natively on CIFAR-10 via `chenyaofo/pytorch-cifar-models`). It achieves **91.4% validation accuracy** in under 3 minutes of fine-tuning on CPU.

The system includes a full **FastAPI web server** with a browser UI, real-time inference, human/text detection and rejection, custom image dataset support, and ultra-fast model retraining (< 5 seconds).

### ✅ Key Features

| Feature | Details |
|---------|---------|
| **Model** | ResNet-20 pre-trained on CIFAR-10 (~91% base accuracy) |
| **Fine-tuning** | 94.7% parameters trained (257,994 / 272,474) |
| **Accuracy** | **91.4% validation** · 91% test |
| **Training Time** | ~3 minutes (CPU) · 10 epochs |
| **Web UI** | FastAPI + HTML/CSS/JS with drag-and-drop |
| **Human Rejection** | Skin-tone + face proportion detection |
| **Text Rejection** | Edge-density heuristic rejects screenshots/docs |
| **Custom Classes** | Add your own images and retrain in **< 5 seconds** |
| **Hot-Swap Retrain** | Model updates in-memory, no server restart needed |

---

## 🏗️ Architecture

```
Input Image → Resize (32×32) → CIFAR-10 Normalization
                                        │
                        ┌───────────────▼────────────────┐
                        │   ResNet-20 (chenyaofo hub)     │
                        │   272,474 total parameters      │
                        │                                 │
                        │  conv1                 frozen   │
                        │  layer1 (BasicBlocks)  frozen   │
                        │  layer2 (BasicBlocks)  trained  │
                        │  layer3 (BasicBlocks)  trained  │
                        │  avgpool               trained  │
                        │  FC (64 → 10)          trained  │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        10-class softmax probabilities
                  (airplane, automobile, ..., truck)

  Phase 1: 257,994 params (94.7%) · 3 epochs · LR=0.0002
  Phase 2: 272,474 params (100%) · 7 epochs · LR=0.00001
```

---

## 📁 Project Structure

```
deep learning/
│
├── train.py                    # Full training & evaluation pipeline
├── infer.py                    # CLI inference script
├── app.py                      # FastAPI web server (inference + retrain API)
├── fast_retrain.py             # Standalone ultra-fast custom retrain script
├── extract_images.py           # Saves CIFAR-10 images to disk as PNG files
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Auto-downloaded CIFAR-10 dataset
│   ├── cifar-10-batches-py/    # Raw CIFAR-10 binary files
│   └── extracted_images/       # PNG images organized by class (10 folders)
│       ├── airplane/
│       ├── automobile/
│       ├── bird/
│       └── ...
│
├── custom_data/                # Your custom images for retraining
│   └── <class_name>/           # e.g. custom_data/dog/my_dog.jpg
│
├── outputs/                    # All generated outputs (auto-created)
│   ├── best_model.pt           # Best model weights (91.4% val accuracy)
│   ├── training_curves.png     # Loss & accuracy curves across 10 epochs
│   ├── confusion_matrix.png    # Per-class confusion matrix heatmap
│   ├── sample_images.png       # Grid of sample CIFAR-10 images
│   ├── training_history.json   # Epoch-by-epoch metrics (used by web UI)
│   └── training_log.txt        # Full training console log
│
└── static/                     # Web UI frontend files
    ├── index.html
    ├── style.css
    └── script.js
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install opencv-python-headless   # For human detection (optional)
```

### 2. Train the model

```bash
python train.py
```

> The CIFAR-10 dataset (~170 MB) and ResNet-20 weights are downloaded automatically on the first run. Training completes in **~3 minutes on CPU**.

### 3. Launch the web server

```bash
python app.py
```

Open your browser at **http://127.0.0.1:8000**

---

## 🚀 Training Pipeline (`train.py`)

### What it does

1. **Downloads CIFAR-10** into `data/` (first run only)
2. **Saves sample images** to `outputs/sample_images.png`
3. **Loads ResNet-20** from `chenyaofo/pytorch-cifar-models` (cached after first download)
4. **Phase 1** — Trains layer2, layer3, and FC head (3 epochs, LR=0.0002)
5. **Phase 2** — Fine-tunes entire network (7 epochs, LR=0.00001)
6. **Saves best model** to `outputs/best_model.pt` based on validation accuracy
7. **Evaluates on test set** with full classification report
8. **Generates outputs:**
   - `outputs/training_curves.png` — Loss & accuracy curves
   - `outputs/confusion_matrix.png` — Confusion matrix heatmap
   - `outputs/training_history.json` — Machine-readable epoch metrics

### Latest Training Results (2026-07-22)

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **91.40%** |
| **Test Accuracy** | **91%** |
| **Total Training Time** | 2m 56s (CPU) |
| **Total Epochs** | 10 (3 + 7) |
| **Batch Size** | 32 |
| **Trainable Params (Ph1)** | 257,994 / 272,474 (94.7%) |

### Per-Class Test Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| ✈️ Airplane | 0.95 | 0.82 | 0.88 |
| 🚗 Automobile | 0.91 | 0.96 | 0.93 |
| 🐦 Bird | 0.86 | 0.89 | **0.87** |
| 🐱 Cat | 0.89 | 0.81 | 0.85 |
| 🦌 Deer | 0.95 | 0.93 | 0.94 |
| 🐕 Dog | 0.82 | 0.90 | **0.86** |
| 🐸 Frog | 0.92 | 0.95 | 0.93 |
| 🐴 Horse | 0.95 | 0.95 | 0.95 |
| 🚢 Ship | 0.95 | 0.99 | **0.97** |
| 🚛 Truck | 0.93 | 0.95 | **0.94** |
| **Average** | **0.91** | **0.91** | **0.91** |

### Training Configuration

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Trainable Layers | layer2 + layer3 + FC | All layers |
| Learning Rate | 0.0002 | 0.00001 |
| Epochs | 3 | 7 |
| Optimizer | SGD + Momentum | SGD + Momentum |
| Loss Function | CrossEntropyLoss (weighted) | CrossEntropyLoss (weighted) |
| Class Weights | dog×3, bird×2.5, truck×2 | dog×3, bird×2.5, truck×2 |

### Data Augmentation Pipeline

| Augmentation | Parameter | Purpose |
|---|---|---|
| `Resize` | 36×36 then crop to 32×32 | Input normalization |
| `RandomCrop` | 32×32, padding=4 | Positional robustness |
| `RandomHorizontalFlip` | p=0.5 | Dataset doubling |
| `RandomRotation` | ±15° | Pose variation (helps dog/bird) |
| `ColorJitter` | brightness/contrast/saturation=0.3 | Lighting robustness |
| `RandomErasing` | p=0.2, scale=(0.02, 0.15) | Occlusion robustness |
| `Normalize` | mean=[0.4914, 0.4822, 0.4465] | CIFAR-10 statistics |

---

## 🔍 CLI Inference (`infer.py`)

Classify a single image from the command line:

```bash
python infer.py --image path/to/image.png
```

**Sample output:**
```
+----------------------------------------------+
|          CIFAR-10 INFERENCE RESULT           |
+----------------------------------------------+
|  Image: airplane_0.png                       |
+------+----------------+-----------------------+
| Rank | Class          | Confidence            |
+------+----------------+-----------------------+
|  1   | airplane       |  99.9% ##################- < |
|  2   | deer           |   0.1% --------------------  |
+------+----------------+-----------------------+

  >> Prediction: AIRPLANE (99.9% confidence)
```

**CLI Options:**

| Flag | Description | Default |
|---|---|---|
| `--image`, `-i` | Path to input image (required) | — |
| `--model`, `-m` | Path to model weights | `outputs/best_model.pt` |

---

## 🌐 Web Server (`app.py`)

### Start the server

```bash
python app.py
```

Visit **http://127.0.0.1:8000** in your browser.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload image → get prediction JSON |
| `POST` | `/add-to-dataset` | Save image under a label for retraining |
| `POST` | `/retrain` | Trigger ultra-fast custom retraining (< 5s) |
| `GET` | `/retrain-status` | Poll retraining progress |
| `GET` | `/training-history` | Epoch metrics for accuracy chart |
| `GET` | `/classes` | List all 10 recognized CIFAR-10 classes |

### Image Rejection System

The server uses a **multi-layer rejection pipeline** before any model inference:

```
Upload Image
     │
     ▼
[Layer 1] Human detection (skin-tone model + OpenCV cascade if available)
     │ Human detected? → NOT RECOGNIZED
     ▼
[Layer 2] Text/document detection (horizontal edge density heuristic)
     │ Text detected? → NOT RECOGNIZED
     ▼
[Layer 3] Model inference (ResNet-20, 32×32, CIFAR-10 normalized)
     │
     ▼
[Layer 4] Entropy check — is the model too uncertain?
     │ entropy_ratio > 0.60 (or 0.85 for custom classes)? → NOT RECOGNIZED
     ▼
[Layer 5] Confidence threshold
     │ confidence < 85% (or 60% for custom classes)? → NOT RECOGNIZED
     ▼
RECOGNIZED — return top-5 predictions
```

---

## 🎨 Custom Dataset & Retraining

### Add a custom image

1. Go to the web UI at **http://127.0.0.1:8000**
2. Upload your image
3. Enter the correct CIFAR-10 class label (e.g., `dog`, `airplane`)
4. Click **"Add to Dataset"**

> **Note:** Labels must be one of the 10 CIFAR-10 classes. The image is saved to `custom_data/<label>/`.

### Retrain on custom data

After adding images, click **"Start Retraining"** in the web UI.

The retrain:
- Freezes the entire ResNet-20 backbone
- Updates only the FC layer (650 parameters)
- Runs **3 epochs** on your custom images
- Completes in **< 5 seconds**
- **Hot-swaps the model in memory** — no server restart required

---

## 📊 CIFAR-10 Classes

| Index | Class | Emoji | Index | Class | Emoji |
|:---:|---|:---:|:---:|---|:---:|
| 0 | Airplane | ✈️ | 5 | Dog | 🐕 |
| 1 | Automobile | 🚗 | 6 | Frog | 🐸 |
| 2 | Bird | 🐦 | 7 | Horse | 🐴 |
| 3 | Cat | 🐱 | 8 | Ship | 🚢 |
| 4 | Deer | 🦌 | 9 | Truck | 🚛 |

**Any image not matching these 10 classes is rejected as "NOT RECOGNIZED".**

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `torchvision` | Pre-trained models & datasets |
| `fastapi` | Web server & REST API |
| `uvicorn` | ASGI server |
| `pillow` | Image loading & processing |
| `numpy` | Numerical operations & detection heuristics |
| `matplotlib` | Training curves & plots |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Classification metrics |
| `opencv-python-headless` | Human face detection (optional) |

Install all:
```bash
pip install -r requirements.txt
pip install opencv-python-headless
```

---

## 📝 Notes

- Training uses a **5,000-sample subset** by default for speed. Pass `--full` to use the complete 50,000-sample dataset.
- The model weights are cached locally after the first download at `~/.cache/torch/hub/`.
- Human detection uses the **Kovac-Peer RGB skin model** as a fallback when OpenCV cascade XML files are unavailable (as in `opencv-python-headless`).
- All operations run in **user-space PyTorch** — no hardware-level or boot-sector access.

---

## 📝 License

This project is for educational and internship demonstration purposes.
