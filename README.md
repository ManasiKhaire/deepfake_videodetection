# 🔍 Deepfake Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF6B6B?logo=gradio&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

**An advanced AI-powered deepfake video detection system using a hybrid deep learning architecture.**

</div>

---

## 📌 About the Project

This project detects AI-generated or manipulated (deepfake) videos using a **hybrid neural network** that combines spatial, temporal, and motion-based features for robust and accurate classification.

Unlike single-model approaches, this system fuses three complementary streams:
- **ResNeXt-50** for extracting deep spatial features from faces
- **Vision Transformer (ViT)** for global attention-based feature understanding
- **Optical Flow CNN** for capturing motion inconsistencies between frames

A user-friendly **Gradio web interface** allows anyone to upload a video and get instant real/fake predictions.

---

## 🧠 Model Architecture

```
Input Video
    │
    ▼
Face Detection (OpenCV Haar Cascade)
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
ResNeXt-50 (Spatial)        Optical Flow CNN (Motion)
[2048-dim features]           [16-dim features]
    │                                  │
    ▼                                  │
ViT (Global Attention)                 │
[768-dim features]                     │
    │                                  │
    └──────────────┬───────────────────┘
                   ▼
         Concatenate [2048 + 768 + 16 = 2832]
                   ▼
           Fully Connected Layers
           [2832 → 512 → 1]
                   ▼
            Sigmoid Output
          (Real / Fake probability)
```

---

## ✨ Features

- 🎯 **Hybrid Detection** — Combines ResNeXt-50 + Vision Transformer + Optical Flow for multi-stream analysis
- 👁️ **Face-Focused Analysis** — Automatically detects and crops faces using OpenCV before processing
- 🌊 **Motion Analysis** — Optical flow captures temporal inconsistencies between consecutive frames
- 🖥️ **Gradio Web UI** — Clean, multi-tab interface with Home, About, and Credits pages
- ⚡ **GPU/CPU Support** — Automatically uses CUDA if available, otherwise falls back to CPU
- 📊 **Confidence Scores** — Outputs probability scores for both Real and Fake classes

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Spatial Features | ResNeXt-50-32x4d (pretrained) |
| Attention Features | Vision Transformer (ViT) via HuggingFace |
| Motion Features | Optical Flow (Farneback) + Custom CNN |
| Face Detection | OpenCV Haar Cascade |
| Web Interface | Gradio |
| Image Processing | OpenCV, torchvision |

---

## 📁 Project Structure

```
deepfake-detector/
│
├── app.py                        # Gradio web app & UI
├── model.py                      # HybridDeepfakeDetector architecture
├── utils.py                      # Video preprocessing & optical flow
├── deepfake_model_weights.pth    # Trained model weights
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure the model weights file is in the root directory
#    deepfake_model_weights.pth

# 4. Run the app
python app.py
```

The Gradio interface will launch in your browser at `http://localhost:7860`

---

## 💡 How It Works

1. **Upload** an MP4 video through the web interface
2. **Face Detection** — OpenCV Haar Cascade locates and crops the face region from each frame
3. **Optical Flow** — Farneback algorithm computes motion vectors between consecutive face frames
4. **Feature Extraction** — ResNeXt-50 extracts spatial CNN features; ViT captures global attention patterns
5. **Fusion** — All three feature streams are concatenated and passed through a classifier
6. **Output** — Probability scores for `Real` and `Fake` are displayed

---

## ⚠️ Limitations & Precautions

- Only **MP4 format** videos are currently supported
- Results should be **verified by experts** in high-stakes or legal situations
- **No detection system is perfect** — false positives/negatives are possible
- Deepfake technology evolves rapidly; the model may not detect all novel manipulation techniques
- Always consider the **source and context** of a video alongside the tool's output

---

## 👥 Team

This project was developed as a final year BE Computer Engineering project.

| Name | Role |
|---|---|
| Manasi Khaire | Developer |
| Pavitra Desai | Developer |
| Sai Nagane | Developer |
| Siddhi Algude | Developer |

**Project Guide:** Prof. Kiran Yesugade

**Institution:** Bharati Vidyapeeth's College of Engineering for Women, Department of Computer Engineering

---

## 📄 License

This project is intended for academic and research purposes only. Please use responsibly and ethically.

---

<div align="center">
Made with ❤️ for combating misinformation
</div>
