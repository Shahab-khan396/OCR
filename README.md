<div align="center">

# 🔍 OCR Studio
### *High-Accuracy Optical Character Recognition Application*

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://python.org)
[![Tesseract OCR](https://img.shields.io/badge/Tesseract-5.x%20(LSTM)-orange?logo=tesseract)](https://github.com/tesseract-ocr/tesseract)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen?logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Built with Streamlit, Tesseract 5 LSTM, OpenCV 4, and Poppler.*  
*Ready for 1-click deployment on **Streamlit Community Cloud**, **Docker**, and **Local environments**.*

---

</div>

## 📌 Overview

**OCR Studio** is a production-ready Optical Character Recognition (OCR) web application built with Streamlit. It extracts text from images and multi-page PDF documents using a configurable multi-stage computer vision preprocessing pipeline combined with Tesseract 5's neural network LSTM engine.

Designed for real-world challenging documents — including phone-photographed receipts, skewed book scans, low-contrast historical documents, invoices, and multi-page PDFs with uneven lighting or shadows.

### ✨ Key Capabilities
- 🚀 **Streamlit Cloud Ready**: Native Debian support via `packages.txt` with cross-platform auto-detection.
- 📄 **Multi-Page PDF Processing**: Fast conversion via Poppler with configurable rendering DPI (200–500 DPI) and progress tracking.
- 🎛️ **Full Preprocessing Pipeline**: Adaptive binarization, auto-deskewing, morphological denoising, shadow elimination, Lanczos upscaling, and border padding.
- 🧠 **Multi-PSM Engine Optimization**: Run multiple Page Segmentation Modes (PSM) in parallel and automatically select the highest accuracy text extraction.
- 🎨 **Confidence Heatmap**: Visual per-word color grading overlaid onto the document (green $\ge$80%, amber 50–79%, red <50%).
- 🩹 **Smart Post-Processing**: Context-aware correction of common OCR optical confusion (e.g., `|` $\rightarrow$ `I`, `0` $\leftrightarrow$ `O`, `l` $\leftrightarrow$ `1`), Unicode NFKC normalization, and noise filtering.
- 💾 **Multi-Format Export**: Export results to plain text (`.txt`), formatted Markdown (`.md`), or preprocessed imagery (`.png`).

---

## 🚀 Deploying to Streamlit Community Cloud

This repository is pre-configured for seamless, zero-config deployment on **[Streamlit Community Cloud](https://share.streamlit.io)**.

### Step 1: Push Repository to GitHub
Ensure all repository files (including `packages.txt`, `requirements.txt`, and `.streamlit/config.toml`) are committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "Configure Streamlit Cloud deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
2. Click **New app**.
3. Fill in the deployment details:
   - **Repository**: `YourUsername/YourRepoName`
   - **Branch**: `main`
   - **Main file path**: `ocr_app.py`
4. Click **Deploy!**

### How Cloud Dependencies Work
Streamlit Community Cloud automatically provisions the required environment using:
- **`packages.txt`**: Installs Linux system packages via `apt-get`:
  - `tesseract-ocr` & `tesseract-ocr-eng`: Tesseract engine and English model.
  - `tesseract-ocr-osd`: Orientation and script detection.
  - `poppler-utils`: PDF rendering tools (`pdfinfo`, `pdftoppm`).
  - `libgl1`: OpenGL runtime for OpenCV.
- **`requirements.txt`**: Installs Python dependencies using `opencv-python-headless` for headless Linux execution without GUI display server issues.
- **`.streamlit/config.toml`**: Configures dark mode aesthetics, increases file upload limit to 100 MB, and sets headless mode.

---

## 🐳 Docker Deployment

You can deploy OCR Studio in any container platform (e.g. Hugging Face Spaces, Google Cloud Run, AWS ECS, Render, Railway, or local Docker):

### Build and Run with Docker
```bash
# Build the Docker image
docker build -t ocr-studio .

# Run the container on port 8501
docker run -d -p 8501:8501 --name ocr-studio ocr-studio
```
Open **`http://localhost:8501`** in your browser.

---

## 💻 Local Installation & Setup

OCR Studio automatically detects Tesseract and Poppler on Windows, macOS, and Linux without requiring manual code modifications.

### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/YourUsername/OCR.git
cd OCR

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Install System Dependencies

#### 🪟 Windows:
1. **Tesseract OCR**:
   - Download the installer from UB-Mannheim: [Tesseract Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to standard path `C:\Program Files\Tesseract-OCR\`. OCR Studio detects this location automatically.
2. **Poppler (for PDF support)**:
   - Download the latest Windows binary release from: [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases)
   - Extract the archive to `C:\poppler\` (such that `C:\poppler\Library\bin\pdfinfo.exe` exists). OCR Studio detects this path automatically.

#### 🐧 Linux (Ubuntu / Debian):
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd poppler-utils libgl1
```

#### 🍎 macOS (Homebrew):
```bash
brew install tesseract poppler
```

### 3. Launch Application
```bash
streamlit run ocr_app.py
```
The app will automatically launch in your browser at `http://localhost:8501`.

---

## ⚙️ Path Auto-Detection & Overrides

OCR Studio uses a cross-platform resolution system that detects binaries in order of precedence:

1. **Environment Variables** (highest priority):
   - `TESSERACT_CMD`: Absolute path to `tesseract` or `tesseract.exe`
   - `POPPLER_PATH`: Path to the directory containing Poppler binaries (`pdfinfo`, `pdftoppm`)
2. **System PATH**:
   - Checks `shutil.which("tesseract")` and `shutil.which("pdfinfo")` (default on Streamlit Cloud, Linux, macOS).
3. **Common Installation Paths**:
   - Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`, `C:\poppler\Library\bin`, etc.
   - Linux/macOS: `/usr/bin/tesseract`, `/usr/local/bin/tesseract`, `/opt/homebrew/bin/tesseract`.

*No hardcoded file editing in `ocr_app.py` is needed!*

---

## 🛠️ Preprocessing & Engine Features

```
Uploaded File (Image or PDF)
   │
   ▼
[ PDF Conversion ] ────────────► Poppler at 200–500 DPI
   │
   ▼
[ Image Adjustments ] ─────────► Brightness, Contrast, Sharpness
   │
   ▼
[ Resolution Check ] ──────────► Lanczos4 Upscaling (if < 2000px)
   │
   ▼
[ Geometry Correction ] ───────► Hough Line Transform Auto-Deskew
   │
   ▼
[ Lighting Normalization ] ────► Morphological Dilation & Median Blur (Shadow Removal)
   │
   ▼
[ Denoising ] ─────────────────► Fast Non-Local Means Filter
   │
   ▼
[ Binarization ] ──────────────► Adaptive / CLAHE+Otsu / Otsu / Unsharp / Sharpen
   │
   ▼
[ Morphological Clean ] ───────► Morph CLOSE (fill gaps) & OPEN (isolate noise)
   │
   ▼
[ Border Padding ] ────────────► 30px Border for Edge Text Detection
   │
   ▼
[ Tesseract 5 OCR ] ───────────► LSTM Neural Engine (Single or Multi-PSM)
   │
   ▼
[ Post-Processing ] ───────────► Unicode NFKC, Context-Aware Substitutions, Junk Removal
   │
   ▼
[ Analytics & Output ] ────────► Text, Markdown, Processed Image, Confidence Heatmap
```

### Preprocessing Modes
| Mode | Description | Best Suited For |
| :--- | :--- | :--- |
| **Adaptive Threshold** | Gaussian-weighted local thresholding | Documents with gradients or uneven lighting |
| **CLAHE + Otsu** | Contrast-Limited Adaptive Histogram Equalization with Otsu | Low-contrast or faded text, camera captures |
| **Otsu Binarize** | Global variance minimization binarization | Clean scans and standard black-and-white pages |
| **Unsharp Mask** | Gaussian blur weighted subtraction | Preserves fine edges on slightly blurry images |
| **Sharpen** | 3×3 convolution kernel sharpening | Soft text outlines |
| **Grayscale** | Direct 8-bit luminance conversion | Clean inputs requiring raw gradient data |

---

## 📋 Recommended Settings Guide

| Document Type | Binarization | Recommended Toggles | Suggested PSM |
| :--- | :--- | :--- | :--- |
| **Printed Document / Book Scan** | Adaptive Threshold | Denoise, Deskew, Upscale, Morph | `PSM 6` (Uniform block) |
| **Smartphone Photo of Text** | CLAHE + Otsu | Remove Shadows, Denoise, Upscale, Deskew | `PSM 3` (Auto) |
| **Receipt / Invoice** | Otsu Binarize | Upscale, Morph Clean, Border Pad | `PSM 6` (Uniform block) |
| **Single Line / Header / Sign** | Adaptive Threshold | Upscale, Morph Clean | `PSM 7` (Single line) |
| **ID Card / Serial Number** | CLAHE + Otsu | Upscale, Morph Clean, Whitelist | `PSM 7` or `PSM 8` |
| **Faded / Low-Contrast Page** | CLAHE + Otsu | Contrast Boost (1.5–2.0), Denoise, Morph | `PSM 6` (Uniform block) |
| **Multi-Column Newspaper / Article**| Adaptive Threshold | Deskew, Upscale, Multi-PSM | `PSM 4` (Single column) or `PSM 3` |

---

## 📂 Project Structure

```
OCR/
├── .streamlit/
│   └── config.toml        # Streamlit theme, headless mode, and upload limits
├── .dockerignore          # Docker build exclusions
├── .gitignore             # Git ignore patterns
├── Dockerfile             # Production container definition for cloud deployment
├── packages.txt           # Debian system packages for Streamlit Cloud (Tesseract, Poppler)
├── requirements.txt       # Python package dependencies (headless OpenCV, Streamlit, etc.)
├── ocr_app.py             # Main Streamlit application
├── LICENSE                # MIT License
└── README.md              # Project documentation and deployment guide
```

---

## ❓ Troubleshooting

### 1. `Tesseract not detected` on Streamlit Cloud
- Check that `packages.txt` exists at the root of your GitHub repository.
- Verify `packages.txt` contains:
  ```txt
  tesseract-ocr
  tesseract-ocr-eng
  ```
- Reboot the Streamlit Cloud app (Manage app $\rightarrow$ Reboot).

### 2. `Poppler not detected / PDF disabled` on Streamlit Cloud
- Ensure `packages.txt` contains `poppler-utils`.
- On Linux, `pdfinfo` and `pdftoppm` are installed to `/usr/bin/`, which OCR Studio auto-detects.

### 3. Missing `libGL.so.1` or OpenCV GUI errors
- `requirements.txt` specifies `opencv-python-headless>=4.8.0.76` and `packages.txt` includes `libgl1`. If you change requirements, make sure to use `opencv-python-headless` rather than `opencv-python`.

### 4. Low Confidence (< 50%) or Missing Characters
- Enable **Remove shadows** to eliminate gradient illumination.
- Switch to **CLAHE + Otsu** and increase the **Contrast** slider (1.5 – 2.0).
- Enable **Upscale** (Tesseract requires $\ge$150–300 DPI for reliable OCR).
- Turn on **Multi-PSM mode** to evaluate multiple page segmentation strategies in parallel.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
