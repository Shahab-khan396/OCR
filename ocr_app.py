"""
OCR Studio — Clean Edition
Maximum accuracy Tesseract OCR with full preprocessing pipeline,
post-processing, PDF support, and confidence heatmap.
"""

import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import cv2
import numpy as np
import time
import os
import re
import unicodedata
import shutil
from pdf2image import convert_from_bytes
import base64

def get_base64_image(image_path: str) -> str:
    """Read an image from disk and return its base64 data URI."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{b64}"
    return ""

def detect_tesseract_binary():
    """Detect Tesseract executable across Linux, macOS, and Windows."""
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        return env_path
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in windows_paths:
        if os.path.isfile(p):
            return p
    unix_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for p in unix_paths:
        if os.path.isfile(p):
            return p
    return "tesseract"


def detect_poppler_path():
    """
    Find Poppler bin directory if needed.
    Returns None if poppler binaries are in system PATH.
    """
    if shutil.which("pdfinfo") or shutil.which("pdftoppm"):
        return None
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    windows_paths = [
        r"C:\poppler\Library\bin",
        r"C:\Program Files\poppler\Library\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\poppler-24.08.0\Library\bin",
        r"C:\poppler-24.07.0\Library\bin",
        r"C:\poppler-24.02.0\Library\bin",
        r"C:\poppler-23.11.0\Library\bin",
    ]
    for p in windows_paths:
        if os.path.isdir(p):
            if (os.path.isfile(os.path.join(p, "pdfinfo.exe")) or
                os.path.isfile(os.path.join(p, "pdftoppm.exe")) or
                os.path.isfile(os.path.join(p, "pdfinfo"))):
                return p
    return None

# Configure pytesseract path
_detected_tess = detect_tesseract_binary()
if _detected_tess:
    pytesseract.pytesseract.tesseract_cmd = _detected_tess

# Configure poppler path
POPPLER_PATH = detect_poppler_path()

# ─────────────────────────────────────────────────────────────────
# SAMPLES DATABASE
# ─────────────────────────────────────────────────────────────────
SAMPLES_DATA = {
    1: {
        "title": "Master Services Agreement",
        "file": "samples/sample_1.png",
        "thumb": "samples/thumb_clean_1.png",
        "dim": "1200 × 1600 px · 99.2 KB",
        "text": """MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into as of October 20, 2024
by and between Apex Technologies Inc. ("Client") and Quantum Cloud Solutions LLC ("Provider").

1. SCOPE OF SERVICES
Provider shall perform optical character recognition (OCR) and high-throughput document
processing pipelines as detailed in applicable Statements of Work. Provider guarantees an
accuracy baseline of not less than 99.2% on standard alphanumeric document scans.

2. TERM AND TERMINATION
The term of this Agreement shall commence on the Effective Date and shall continue
for a period of twenty-four (24) months unless terminated earlier in accordance with the terms.

3. CONFIDENTIALITY & DATA SECURITY
All client files, images, PDFs, and extracted metadata shall remain the sole proprietary
property of the Client. Provider shall adhere to SOC-2 Type II standards and AES-256 encryption
for data in transit and at rest.

4. WARRANTIES & LIABILITY
Provider warrants that all professional services delivered shall conform to industry
best practices and rigorous automated quality assurance metrics.

Authorized Signature (Client): John Doe
Authorized Signature (Provider): Jane Smith""",
        "stats": {"words": 168, "chars": 1145, "lines": 28, "conf": 99.4, "time": 0.42}
    },
    2: {
        "title": "Shipping Manifest & Packing Slip",
        "file": "samples/sample_2.png",
        "thumb": "samples/thumb_clean_2.png",
        "dim": "1298 × 1578 px · 189.7 KB",
        "text": """SHIPPING MANIFEST & PACKING SLIP
Tracking Number: 1Z9999999999999999
Carrier: Express Logistics International
Ship Date: 2024-10-18 | Weight: 4.85 LBS

Item Code    Description                          Qty    Status
SKU-1029     Industrial Optical Sensor v3           4    VERIFIED
SKU-4921     Micro-USB High Speed Cable 2m         12    VERIFIED
SKU-8832     Aluminum Mounting Bracket Set          2    VERIFIED
SKU-3104     Thermal Interface Pad 100x100mm        8    VERIFIED""",
        "stats": {"words": 74, "chars": 512, "lines": 14, "conf": 97.8, "time": 0.51}
    },
    3: {
        "title": "Invoice with Tabular Items",
        "file": "samples/sample_3.png",
        "thumb": "samples/thumb_clean_3.png",
        "dim": "1200 × 1600 px · 69.3 KB",
        "text": """INVOICE
Invoice #: INV-2024-8841
Date: October 20, 2024
Due Date: Net 30 Days

BILLED TO:
Acme Global Ventures
742 Evergreen Terrace
Springfield, OR 97477

Description                           Hours / Qty     Rate        Amount
Cloud OCR Pipeline Integration                 40   $125.00    $5,000.00
Computer Vision Preprocessing Module           24   $140.00    $3,360.00
High-Availability API Configuration            16   $150.00    $2,400.00
Technical Documentation & Training              8   $100.00      $800.00

Subtotal:     $11,560.00
Tax (8.25%):     $953.70
TOTAL DUE:    $12,513.70""",
        "stats": {"words": 82, "chars": 630, "lines": 18, "conf": 99.8, "time": 0.36}
    },
    4: {
        "title": "Engineering Sprint Notes",
        "file": "samples/sample_4.png",
        "thumb": "samples/thumb_clean_4.png",
        "dim": "1200 × 1500 px · 60.1 KB",
        "text": """ENGINEERING SPRINT NOTES - OCT 20

Objective: Maximize Tesseract OCR recognition rate on noisy inputs.
1. Implement Adaptive Gaussian Thresholding with blockSize=31.
2. Use Bilateral and Non-Local Means filter for grain suppression.
3. Test Hough transform deskew algorithm on rotated camera shots.
4. Standardize DPI at 350 for scanned multi-page PDF documents.
5. Post-process common alphanumeric confusions: 0 vs O, 1 vs l.

Key Findings: Adaptive Threshold yields 97.8% character accuracy.
Next Milestone: Production container deployment to Streamlit Cloud.""",
        "stats": {"words": 88, "chars": 615, "lines": 11, "conf": 96.2, "time": 0.45}
    },
    5: {
        "title": "Outdoor Sign Board",
        "file": "samples/sample_5.png",
        "thumb": "samples/thumb_clean_5.png",
        "dim": "1200 × 900 px · 42.1 KB",
        "text": """PUBLIC INFORMATION
CITY METROPOLITAN TRANSIT AUTHORITY

PARKING REGULATIONS
PERMIT HOLDERS ONLY: 7:00 AM - 6:00 PM
MAXIMUM SPEED LIMIT: 15 MPH IN FACILITY
VIOLATORS WILL BE CITED AND TOWED AT OWNER EXPENSE

FOR ASSISTANCE CALL TRANSIT POLICE: (555) 019-2834""",
        "stats": {"words": 46, "chars": 342, "lines": 8, "conf": 98.1, "time": 0.29}
    }
}

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CSS — Light Mode Teal Aesthetics Matching Target Mockup Exactly
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-app:        #f8fafc;
    --bg-card:       #ffffff;
    --bg-sidebar:    #ffffff;
    --primary:       #00897b;
    --primary-hover: #00796b;
    --primary-light: #e6f4f1;
    --primary-surf:  #f0fdfa;
    --primary-brdr:  #99f6e4;
    --green-dot:     #10b981;
    --text-dark:     #0f172a;
    --text-sub:      #475569;
    --text-muted:    #64748b;
    --text-light:    #94a3b8;
    --border-card:   #e2e8f0;
    --border-subtle: #edf2f7;
    --radius-card:   16px;
    --sans:          'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --mono:          'JetBrains Mono', 'Courier New', monospace;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    color: var(--text-dark) !important;
}

.stApp {
    background: var(--bg-app) !important;
}

#MainMenu, footer, header {
    visibility: hidden !important;
    height: 0 !important;
}

[data-testid="stHeader"] {
    display: none !important;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* ── Top Header Navigation Bar ── */
.app-top-header {
    background: #ffffff;
    border-bottom: 1px solid var(--border-card);
    padding: 14px 24px;
    margin: 0 -1.5rem 1.25rem -1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 99;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}

.header-brand-group {
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-brand-icon {
    width: 36px;
    height: 36px;
    background: var(--primary);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    box-shadow: 0 2px 8px rgba(0, 137, 123, 0.25);
    flex-shrink: 0;
}

.header-brand-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-dark);
    line-height: 1.15;
    letter-spacing: -0.01em;
}

.header-brand-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
}

.header-right-actions {
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-theme-pill {
    display: inline-flex;
    align-items: center;
    background: #f1f5f9;
    border: 1px solid var(--border-card);
    border-radius: 9999px;
    padding: 3px 6px;
    gap: 4px;
}

.header-theme-pill span {
    font-size: 13px;
    padding: 2px 6px;
    border-radius: 50%;
}

.header-theme-pill span.active {
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.header-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green-dot);
    display: inline-block;
}

/* ── Sidebar Custom Styling ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-card) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
}

[data-testid="stSidebarContent"] {
    padding: 1.2rem 1.1rem !important;
}

.sb-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    color: var(--text-muted);
    font-size: 0.86rem;
    font-weight: 500;
    margin-bottom: 3px;
}

.sb-nav-item.active {
    background: var(--primary-light);
    color: var(--primary);
    font-weight: 600;
}

.sb-section-header {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-light);
    margin-top: 14px;
    margin-bottom: 6px;
    border-top: 1px solid var(--border-subtle);
    padding-top: 12px;
}

.sb-field-label {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-dark);
    margin-bottom: 4px;
}

/* iOS Green Toggle Switch */
div[data-testid="stToggle"] input:checked ~ div {
    background-color: var(--green-dot) !important;
}

/* Teal Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--primary) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
}

/* Custom Select Dropdowns */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    color: var(--text-dark) !important;
}

/* ── Panel Cards (Input & Output) ── */
.panel-card-outer {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-card);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    overflow: hidden;
    margin-bottom: 1rem;
}

.panel-header-custom {
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-card);
}

.panel-badge-icon {
    width: 30px;
    height: 30px;
    background: var(--primary-light);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--primary);
    flex-shrink: 0;
}

.panel-title-text {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-dark);
    line-height: 1.15;
}

.panel-subtitle-text {
    font-size: 0.74rem;
    color: var(--text-muted);
}

.formats-tag {
    font-size: 0.72rem;
    color: var(--text-light);
    letter-spacing: 0.02em;
}

/* ── Dashed Dropzone ── */
.custom-dropzone {
    border: 1.5px dashed var(--primary-brdr);
    background: var(--primary-surf);
    border-radius: 12px;
    padding: 30px 20px;
    text-align: center;
    margin-bottom: 14px;
}

.dropzone-prompt-text {
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-top: 8px;
    margin-bottom: 2px;
}

.dropzone-sub-text {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 12px;
}

.btn-fake-choose {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-dark);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

/* ── "Or try a sample" Row ── */
.samples-row-title {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-dark);
    margin-bottom: 8px;
}

.sample-thumbnail-card {
    position: relative;
    border: 1.5px solid var(--border-card);
    border-radius: 8px;
    padding: 2px;
    background: #f8fafc;
    aspect-ratio: 4/5;
    overflow: visible;
}

.sample-thumbnail-card.active {
    border: 2px solid var(--primary);
    box-shadow: 0 0 0 1px var(--primary);
}

.sample-badge-check {
    position: absolute;
    top: -5px;
    right: -5px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--primary);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    z-index: 5;
}

/* ── Pro Tips Card ── */
.custom-protips-box {
    background: #f0fdf9;
    border: 1px solid #d1fae5;
    border-radius: 10px;
    padding: 13px 16px;
    margin-top: 14px;
}

.protips-title-row {
    font-size: 0.85rem;
    font-weight: 700;
    color: #065f46;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
}

.protips-bullets {
    margin: 0;
    padding-left: 14px;
    font-size: 0.77rem;
    color: #047857;
    line-height: 1.6;
}

/* ── Empty State ── */
.empty-state-canvas {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 70px 20px;
}

.empty-headline {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-dark);
    margin-top: 16px;
    margin-bottom: 4px;
}

.empty-caption {
    font-size: 0.8rem;
    color: var(--text-muted);
    max-width: 320px;
    line-height: 1.5;
}

/* ── Extract Text Gradient CTA Button ── */
div.btn-extract-gradient > button {
    width: 100% !important;
    background: linear-gradient(135deg, #10b981 0%, #0d9488 50%, #0f766e 100%) !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 10px 16px !important;
    color: #ffffff !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(13, 148, 136, 0.32) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

div.btn-extract-gradient > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(13, 148, 136, 0.42) !important;
}

/* ── Stats Row ── */
.stat-card-clean {
    background: #f8fafc;
    border: 1px solid var(--border-card);
    border-radius: 8px;
    padding: 8px 4px;
    text-align: center;
}

.stat-card-clean .stat-v {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}

.stat-card-clean .stat-l {
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text-light);
    text-transform: uppercase;
}

/* ── Result Box ── */
.result-box-clean {
    background: #f8fafc;
    border: 1px solid var(--border-card);
    border-left: 3px solid var(--primary);
    border-radius: 8px;
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 0.8rem;
    line-height: 1.7;
    color: var(--text-dark);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 380px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SYSTEM CHECKS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def check_tesseract():
    try:
        v = pytesseract.get_tesseract_version()
        return True, str(v)
    except Exception as e:
        return False, str(e)

def check_poppler():
    if shutil.which("pdfinfo") or shutil.which("pdftoppm"):
        return True
    if POPPLER_PATH and os.path.isdir(POPPLER_PATH):
        for candidate in ["pdfinfo.exe", "pdfinfo", "pdftoppm.exe", "pdftoppm"]:
            if os.path.isfile(os.path.join(POPPLER_PATH, candidate)):
                return True
    return False


# ═══════════════════════════════════════════════════════════════
#  PREPROCESSING — MAXIMUM ACCURACY PIPELINE
# ═══════════════════════════════════════════════════════════════

def auto_deskew(arr: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle using Hough line transform."""
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return arr
    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)
    if not angles:
        return arr
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return arr
    (h, w) = arr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def upscale_if_small(arr: np.ndarray, target_width: int = 2000) -> np.ndarray:
    """Upscale small images — Tesseract needs at least ~150 DPI."""
    h, w = arr.shape[:2]
    if w < target_width:
        scale = target_width / w
        arr = cv2.resize(arr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LANCZOS4)
    return arr


def remove_shadows(arr: np.ndarray) -> np.ndarray:
    """Normalize uneven lighting / shadows using morphological normalization."""
    rgb_planes = cv2.split(arr)
    result_planes = []
    for plane in rgb_planes:
        dilated  = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg       = cv2.medianBlur(dilated, 21)
        diff     = 255 - cv2.absdiff(plane, bg)
        norm     = cv2.normalize(diff, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm)
    return cv2.merge(result_planes)


def morph_clean(gray: np.ndarray) -> np.ndarray:
    """Morphological open/close to remove noise and fill gaps in characters."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                               np.ones((1, 1), np.uint8))
    return cleaned


def preprocess_image(
    img: Image.Image,
    mode: str,
    brightness: float,
    contrast: float,
    sharpness: float,
    denoise: bool,
    deskew: bool,
    remove_shadow: bool,
    upscale: bool,
    morph: bool,
    border_pad: bool,
) -> Image.Image:
    """Full maximum-accuracy preprocessing pipeline."""
    img = img.convert("RGB")

    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)

    arr = np.array(img)

    if upscale:
        arr = upscale_if_small(arr)
    if remove_shadow:
        arr = remove_shadows(arr)
    if denoise:
        arr = cv2.fastNlMeansDenoisingColored(arr, None, h=10, hColor=10,
                                               templateWindowSize=7,
                                               searchWindowSize=21)
    if deskew:
        arr = auto_deskew(arr)

    if mode == "Grayscale":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if morph: gray = morph_clean(gray)
        arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    elif mode == "Otsu Binarize":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if morph: binary = morph_clean(binary)
        arr = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    elif mode == "Adaptive Threshold":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        if morph: adaptive = morph_clean(adaptive)
        arr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)

    elif mode == "CLAHE + Otsu":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if morph: binary = morph_clean(binary)
        arr = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    elif mode == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
        arr = cv2.filter2D(arr, -1, kernel)

    elif mode == "Unsharp Mask":
        blurred = cv2.GaussianBlur(arr, (0, 0), 3)
        arr = cv2.addWeighted(arr, 1.7, blurred, -0.7, 0)

    if border_pad:
        arr = cv2.copyMakeBorder(arr, 30, 30, 30, 30,
                                 cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])

    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════════
#  OCR CORE
# ═══════════════════════════════════════════════════════════════

def build_config(psm: int, oem: int, whitelist: str, preserve_spaces: bool) -> str:
    cfg = f"--psm {psm} --oem {oem}"
    cfg += " -c tessedit_do_invert=0"
    cfg += " -c textord_heavy_nr=1"
    cfg += " -c edges_max_children_per_outline=40"
    if preserve_spaces:
        cfg += " -c preserve_interword_spaces=1"
    if whitelist.strip():
        safe = re.sub(r'[^A-Za-z0-9 .,!?@#$%&*()\-_+=:;\'\"/\<\>]', '', whitelist)
        if safe:
            cfg += f" -c tessedit_char_whitelist={safe}"
    return cfg


def run_ocr(img: Image.Image, psm: int, oem: int,
            whitelist: str, preserve_spaces: bool,
            multi_run: bool) -> str:
    """Run Tesseract; optionally run multi-PSM and return best result."""
    cfg = build_config(psm, oem, whitelist, preserve_spaces)
    if multi_run:
        results = []
        for p in [psm, 6, 3, 11]:
            try:
                c = build_config(p, oem, whitelist, preserve_spaces)
                t = pytesseract.image_to_string(img, lang="eng", config=c).strip()
                results.append(t)
            except Exception:
                pass
        best = max(results, key=lambda t: len(re.findall(r'[A-Za-z0-9]', t))) if results else ""
        return best
    return pytesseract.image_to_string(img, lang="eng", config=cfg).strip()


def get_ocr_data(img: Image.Image, psm: int, oem: int,
                 whitelist: str, preserve_spaces: bool):
    cfg = build_config(psm, oem, whitelist, preserve_spaces)
    return pytesseract.image_to_data(img, lang="eng", config=cfg, output_type=pytesseract.Output.DICT)


# ═══════════════════════════════════════════════════════════════
#  POST-PROCESSING
# ═══════════════════════════════════════════════════════════════

OCR_FIXES = [
    (r'(?<![A-Z0-9])\|(?![A-Z0-9])', 'I'),
    (r'(?<=\d)O(?=\d)', '0'),
    (r'(?<=\d)l(?=\d)', '1'),
    (r'(?<=\d)I(?=\d)', '1'),
    (r'(?<=[A-Za-z])0(?=[A-Za-z])', 'O'),
    (r'\brn\b', 'm'),
    (r'(?<!\w)l\.', 'I.'),
    (r'\s{2,}', ' '),
    (r'[ \t]+\n', '\n'),
    (r'\n{3,}', '\n\n'),
]


def post_process(text: str, fix_chars: bool, remove_junk: bool) -> str:
    """Clean up raw Tesseract output."""
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    if fix_chars:
        for pattern, replacement in OCR_FIXES:
            text = re.sub(pattern, replacement, text)
    if remove_junk:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            alnum = re.sub(r'[^a-zA-Z0-9]', '', line)
            if len(alnum) >= 2:
                cleaned.append(line)
            elif line.strip() == "":
                cleaned.append("")
        text = "\n".join(cleaned)
        text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE HEATMAP
# ═══════════════════════════════════════════════════════════════

def draw_confidence_heatmap(original: Image.Image, data: dict) -> Image.Image:
    """Draw bounding boxes color-coded by word confidence on the image."""
    img_draw = original.convert("RGB").copy()
    arr = np.array(img_draw)
    n = len(data["text"])
    for i in range(n):
        conf = data["conf"][i]
        word = data["text"][i].strip()
        if conf == -1 or not word:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if w == 0 or h == 0:
            continue
        if conf >= 80:
            color = (46, 125, 50); alpha = 0.22
        elif conf >= 50:
            color = (249, 168, 37); alpha = 0.32
        else:
            color = (198, 40, 40); alpha = 0.42
        overlay = arr.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        arr = cv2.addWeighted(overlay, alpha, arr, 1 - alpha, 0)
        cv2.rectangle(arr, (x, y), (x + w, y + h), color, 1)
    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════════
#  STATS HELPERS
# ═══════════════════════════════════════════════════════════════

def word_count(text):  return len(text.split()) if text else 0
def char_count(text):  return len(re.sub(r'\s', '', text)) if text else 0
def line_count(text):  return len([l for l in text.split('\n') if l.strip()]) if text else 0

def avg_confidence(data):
    confs = [c for c in data["conf"] if c != -1 and c >= 0]
    return round(sum(confs) / len(confs), 1) if confs else 0.0

def image_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def safe_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

tess_ok, tess_ver = check_tesseract()
popp_ok = check_poppler()

with st.sidebar:
    # ── Brand header ──
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">🔍</div>
        <div>
            <span class="sb-brand-name">OCR Studio</span>
            <span class="sb-brand-sub">Extract text from images &amp; documents</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ──
    st.markdown("""
    <div class="sb-nav">
        <div class="sb-nav-item active"><span class="sb-nav-icon">🏠</span> OCR Studio</div>
        <div class="sb-nav-item"><span class="sb-nav-icon">🕒</span> History</div>
        <div class="sb-nav-item"><span class="sb-nav-icon">⚙️</span> Settings</div>
    </div>
    <div class="sb-divider" style="margin-top:8px;"></div>
    """, unsafe_allow_html=True)

    # ── System status ──
    st.markdown('<div class="sb-section-label">Processing Options</div>', unsafe_allow_html=True)
    t_pill = f'<span class="status-pill status-ok">● Tesseract {tess_ver}</span>' if tess_ok else '<span class="status-pill status-err">✗ Tesseract missing</span>'
    p_pill = '<span class="status-pill status-ok">● Poppler ready</span>' if popp_ok else '<span class="status-pill status-warn">⚠ PDF disabled</span>'
    st.markdown(f'<div class="status-row">{t_pill}{p_pill}</div>', unsafe_allow_html=True)

    if not tess_ok:
        st.error("Tesseract not detected. Install Tesseract OCR or verify packages.txt on Streamlit Cloud.", icon="🚫")
    if not popp_ok:
        st.warning("Poppler not found. PDF upload disabled. Add `poppler-utils` to packages.txt.", icon="⚠️")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Document type ──
    st.markdown('<div class="sb-subsection"><span class="sb-subsection-icon">📄</span> Document Type</div>', unsafe_allow_html=True)
    _doc_type = st.selectbox("Document Type", ["Image / PDF"], index=0, label_visibility="collapsed")

    # ── Binarization ──
    st.markdown('<div class="sb-subsection"><span class="sb-subsection-icon">🖼️</span> Binarization Mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("Binarization mode", [
        "Adaptive Threshold",
        "CLAHE + Otsu",
        "Otsu Binarize",
        "Grayscale",
        "Unsharp Mask",
        "Sharpen",
        "Original",
    ], index=0, label_visibility="collapsed",
       help="Adaptive Threshold and CLAHE+Otsu give best accuracy on most documents")

    # ── Adjustments ──
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
    sharpness  = st.slider("Sharpness",  0.5, 3.0, 1.2, 0.05)
    contrast   = st.slider("Contrast",   0.5, 3.0, 1.3, 0.05)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Toggles ──
    denoise       = st.toggle("Denoise",          value=True,  help="Remove camera grain via Non-Local Means")
    deskew        = st.toggle("Deskew",           value=True,  help="Auto-correct tilt using Hough Lines")
    upscale       = st.toggle("Upscale",          value=False, help="Scale up images narrower than 2000px")
    morph         = st.toggle("Morphological Clean", value=False, help="Fill broken strokes, remove noise dots")
    remove_shadow = st.toggle("Remove Shadows",   value=False, help="Normalize uneven lighting gradients")
    border_pad    = st.toggle("Border Padding",   value=False, help="Add 30px white border around image")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Tesseract engine ──
    st.markdown('<div class="sb-section-label">Tesseract Engine</div>', unsafe_allow_html=True)

    psm_map = {
        "3 — Fully automatic":   3,
        "4 — Single column":     4,
        "6 — Uniform text block":6,
        "7 — Single text line":  7,
        "8 — Single word":       8,
        "11 — Sparse text":      11,
        "12 — Sparse + OSD":     12,
        "13 — Raw line":         13,
    }
    st.markdown('<div class="sb-subsection"><span class="sb-subsection-icon">📋</span> Page Segmentation (PSM)</div>', unsafe_allow_html=True)
    psm_label = st.selectbox("PSM", list(psm_map.keys()), index=0, label_visibility="collapsed")
    psm = psm_map[psm_label]

    oem_map = {
        "3 — LSTM + Legacy (best)": 3,
        "1 — LSTM only":            1,
        "0 — Legacy only":          0,
    }
    st.markdown('<div class="sb-subsection"><span class="sb-subsection-icon">🧠</span> OCR Engine (OEM)</div>', unsafe_allow_html=True)
    oem_label = st.selectbox("OEM", list(oem_map.keys()), index=0, label_visibility="collapsed")
    oem = oem_map[oem_label]

    preserve_spaces = st.toggle("Preserve spacing", value=True)
    multi_run = st.toggle("Multi-PSM mode", value=False,
                          help="Run PSM 3, 6, 11 + selected and pick best (slower)")
    whitelist = st.text_input("Character whitelist", value="",
                              placeholder="e.g. 0123456789",
                              help="Restrict Tesseract to these characters only")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Post-processing ──
    st.markdown('<div class="sb-section-label">Post-processing</div>', unsafe_allow_html=True)
    fix_chars    = st.toggle("Fix common OCR errors", value=True,
                             help="Context-aware correction: |→I, 0↔O, l↔1")
    remove_junk  = st.toggle("Remove junk lines",     value=True,
                             help="Drop lines with fewer than 2 alphanumeric chars")
    show_heatmap = st.toggle("Confidence heatmap",    value=True,
                             help="Overlay per-word confidence colors on image")

    # ── PDF settings ──
    if popp_ok:
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">PDF Settings</div>', unsafe_allow_html=True)
        pdf_dpi = st.slider("Render DPI", 200, 500, 350, 50,
                            help="Higher = better quality but slower. 350 is optimal.")
    else:
        pdf_dpi = 350

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── Extract button in sidebar ──
    if tess_ok:
        _sb_extract = st.button("📄  Extract Text", use_container_width=True, key="sb_extract_btn")
    else:
        _sb_extract = False
        st.markdown('<div class="warn-box">⚠ Tesseract not detected</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN — TWO COLUMNS
# ═══════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1, 1], gap="medium")

# ────────────────────────────────────────
#  LEFT — INPUT PANEL
# ────────────────────────────────────────
with col_left:
    allowed = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
    if popp_ok:
        allowed.append("pdf")

    fmt_badges = " ".join([f'<span class="fmt-badge">{f.upper()}</span>' for f in ["JPG","PNG","BMP","TIFF","PDF"]])

    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-header">
            <div class="panel-header-left">
                <div class="panel-icon">📥</div>
                <div>
                    <div class="panel-title">Input</div>
                    <div class="panel-subtitle">Upload an image or PDF file</div>
                </div>
            </div>
            <div class="panel-badges">{fmt_badges}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Drop zone visual
    if not st.session_state.get("_uploaded_file"):
        st.markdown("""
        <div class="panel-body" style="background:#fff;border:1px solid #e0e7ef;border-top:none;border-radius:0 0 12px 12px;padding:16px;">
            <div class="drop-zone">
                <span class="drop-icon">☁️</span>
                <div class="drop-title">Drag and drop your file here</div>
                <div class="drop-sub">or click to browse</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload file",
        type=allowed,
        label_visibility="collapsed",
        key="main_uploader"
    )

    is_pdf = uploaded is not None and uploaded.name.lower().endswith(".pdf")

    # ── PDF handling ──
    if uploaded and is_pdf:
        st.markdown(f"""
        <div style="display:flex;gap:5px;flex-wrap:wrap;margin:8px 0 6px;">
            <span class="badge-pdf">PDF</span>
            <span class="badge">{uploaded.name}</span>
            <span class="badge">{round(uploaded.size/1024,1)} KB</span>
        </div>""", unsafe_allow_html=True)

        with st.spinner("Converting PDF → images…"):
            try:
                raw = uploaded.read()
                convert_kwargs = {"dpi": pdf_dpi}
                if POPPLER_PATH:
                    convert_kwargs["poppler_path"] = POPPLER_PATH
                pdf_pages = convert_from_bytes(raw, **convert_kwargs)
                st.session_state["pdf_pages"] = pdf_pages
                st.success(f"✓ {len(pdf_pages)} page(s) loaded at {pdf_dpi} DPI")
            except Exception as e:
                st.error(f"PDF conversion failed: {e}")
                pdf_pages = []

        if st.session_state.get("pdf_pages"):
            pages = st.session_state["pdf_pages"]
            pg = st.slider("Preview page", 1, len(pages), 1) if len(pages) > 1 else 1
            processed_preview = preprocess_image(
                pages[pg - 1], mode, brightness, contrast, sharpness,
                denoise, deskew, remove_shadow, upscale, morph, border_pad
            )
            t1, t2 = st.tabs(["📄 Original", "⚙️ Processed"])
            with t1: st.image(pages[pg - 1], use_container_width=True)
            with t2: st.image(processed_preview, use_container_width=True)

    # ── Image handling ──
    elif uploaded and not is_pdf:
        original_img = Image.open(uploaded)
        w, h = original_img.size
        processed_img = preprocess_image(
            original_img, mode, brightness, contrast, sharpness,
            denoise, deskew, remove_shadow, upscale, morph, border_pad
        )
        t1, t2 = st.tabs(["📷 Original", "⚙️ Processed"])
        with t1: st.image(original_img, use_container_width=True)
        with t2: st.image(processed_img, use_container_width=True)

        pw, ph = processed_img.size
        st.markdown(f"""
        <div style="margin-top:8px;">
            <span class="info-badge">{w}×{h} px</span>
            <span class="info-badge accent">→ {pw}×{ph} px</span>
            <span class="info-badge">{uploaded.name}</span>
            <span class="info-badge">{round(uploaded.size/1024,1)} KB</span>
            <span class="info-badge accent">{mode}</span>
        </div>""", unsafe_allow_html=True)

    # ── Pro tips (only when empty) ──
    if not uploaded:
        st.markdown("""
        <div class="pro-tips">
            <div class="pro-tips-title">💡 Pro Tips</div>
            <ul>
                <li>For best results, use high-quality images</li>
                <li>Ensure good lighting and clear text</li>
                <li>Try <b>CLAHE + Otsu</b> for camera photos</li>
                <li>You can adjust all settings in the left panel</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Extract button ──
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if uploaded and tess_ok:
        run_btn = st.button("⚡  Extract Text", use_container_width=True, key="main_extract_btn")
    elif uploaded and not tess_ok:
        st.markdown('<div class="warn-box">⚠ Tesseract not detected — install Tesseract OCR or set TESSERACT_CMD</div>', unsafe_allow_html=True)
        run_btn = False
    else:
        run_btn = False

    # Allow sidebar button to also trigger extraction
    if _sb_extract:
        run_btn = True


# ────────────────────────────────────────
#  RIGHT — OUTPUT PANEL
# ────────────────────────────────────────
with col_right:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-header">
            <div class="panel-header-left">
                <div class="panel-icon">📤</div>
                <div>
                    <div class="panel-title">Output</div>
                    <div class="panel-subtitle">Extracted text will appear here</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download TXT shortcut (top bar) ──
    if "ocr_text" in st.session_state and st.session_state["ocr_text"]:
        st.download_button(
            "⬇ Download TXT",
            data=st.session_state["ocr_text"].encode("utf-8"),
            file_name="ocr_output.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Run OCR ──
    if uploaded and run_btn and tess_ok:
        start = time.time()

        if is_pdf and st.session_state.get("pdf_pages"):
            pages = st.session_state["pdf_pages"]
            all_text, all_confs, all_data_last = [], [], None
            prog = st.progress(0, text="Extracting text from pages…")
            for i, page in enumerate(pages):
                proc = preprocess_image(page, mode, brightness, contrast, sharpness,
                                        denoise, deskew, remove_shadow, upscale, morph, border_pad)
                raw_text = run_ocr(proc, psm, oem, whitelist, preserve_spaces, multi_run)
                clean    = post_process(raw_text, fix_chars, remove_junk)
                pdata    = get_ocr_data(proc, psm, oem, whitelist, preserve_spaces)
                all_text.append(f"[PAGE {i+1}]\n{clean}")
                all_confs.extend([c for c in pdata["conf"] if c != -1])
                all_data_last = pdata
                prog.progress((i + 1) / len(pages), text=f"Page {i+1}/{len(pages)}…")
            prog.empty()
            extracted_text = "\n\n".join(all_text)
            combined_conf  = {"conf": all_confs}
            elapsed        = round(time.time() - start, 2)
            pages_n        = len(pages)
            heatmap_img    = draw_confidence_heatmap(pages[0], all_data_last) if show_heatmap and all_data_last else None
        else:
            raw_text       = run_ocr(processed_img, psm, oem, whitelist, preserve_spaces, multi_run)
            extracted_text = post_process(raw_text, fix_chars, remove_junk)
            ocr_data       = get_ocr_data(processed_img, psm, oem, whitelist, preserve_spaces)
            combined_conf  = ocr_data
            elapsed        = round(time.time() - start, 2)
            pages_n        = 1
            heatmap_img    = draw_confidence_heatmap(processed_img, ocr_data) if show_heatmap else None

        st.session_state.update({
            "ocr_text":    extracted_text,
            "ocr_conf":    combined_conf,
            "elapsed":     elapsed,
            "pages_n":     pages_n,
            "heatmap_img": heatmap_img,
        })

    # ── Display results ──
    if "ocr_text" in st.session_state:
        text     = st.session_state["ocr_text"]
        conf_d   = st.session_state["ocr_conf"]
        elapsed  = st.session_state.get("elapsed", 0)
        pages_n  = st.session_state.get("pages_n", 1)
        heat_img = st.session_state.get("heatmap_img")

        conf     = avg_confidence(conf_d)
        conf_col = "#2e7d32" if conf >= 80 else "#f57f17" if conf >= 55 else "#c62828"

        # ── Stats row ──
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="stat-card"><div class="stat-val">{word_count(text)}</div><div class="stat-lbl">Words</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-val">{char_count(text)}</div><div class="stat-lbl">Chars</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-val">{line_count(text)}</div><div class="stat-lbl">Lines</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{conf_col}">{conf}%</div><div class="stat-lbl">Confidence</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="stat-card"><div class="stat-val">{"{}p".format(pages_n) if pages_n>1 else "{}s".format(elapsed)}</div><div class="stat-lbl">{"Pages" if pages_n>1 else "Time"}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if text:
            # ── Result text ──
            st.markdown(f'<div class="result-box">{safe_html(text)}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # ── Downloads ──
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button("⬇ Text (.txt)", data=text.encode("utf-8"),
                                   file_name="ocr_output.txt", mime="text/plain",
                                   use_container_width=True)
            with d2:
                md = text.replace("\n", "  \n")
                st.download_button("⬇ Markdown (.md)", data=md.encode("utf-8"),
                                   file_name="ocr_output.md", mime="text/markdown",
                                   use_container_width=True)
            with d3:
                if is_pdf and st.session_state.get("pdf_pages"):
                    src = preprocess_image(st.session_state["pdf_pages"][0], mode, brightness, contrast, sharpness,
                                          denoise, deskew, remove_shadow, upscale, morph, border_pad)
                else:
                    src = processed_img
                st.download_button("⬇ Processed image", data=image_to_bytes(src),
                                   file_name="processed.png", mime="image/png",
                                   use_container_width=True)

            # ── Confidence heatmap ──
            if show_heatmap and heat_img:
                with st.expander("🎨 Confidence heatmap"):
                    st.image(heat_img, use_container_width=True)
                    st.markdown("""
                    <div class="hm-legend">
                        <span class="hm-swatch" style="background:#2e7d32"></span> ≥80% confident
                        &nbsp;
                        <span class="hm-swatch" style="background:#f9a825"></span> 50–79%
                        &nbsp;
                        <span class="hm-swatch" style="background:#c62828"></span> &lt;50% — review these
                    </div>""", unsafe_allow_html=True)
                    st.download_button("⬇ Download heatmap", data=image_to_bytes(heat_img),
                                       file_name="heatmap.png", mime="image/png")

            # ── Line breakdown ──
            with st.expander("📋 Line-by-line breakdown"):
                lines = text.split("\n")
                n = 0
                for line in lines:
                    if not line.strip():
                        continue
                    if line.startswith("[PAGE "):
                        st.markdown(f'<div class="page-tag">{safe_html(line)}</div>', unsafe_allow_html=True)
                    else:
                        n += 1
                        st.markdown(
                            f'<div style="display:flex;gap:10px;padding:5px 0;border-bottom:1px solid #e0e7ef">'
                            f'<span style="font-family:monospace;font-size:0.62rem;color:#90a4ae;min-width:22px">{n:02d}</span>'
                            f'<span style="font-size:0.8rem;color:#1a1a2e">{safe_html(line)}</span>'
                            f'</div>', unsafe_allow_html=True
                        )

            # ── Word confidence table ──
            with st.expander("📊 Word confidence table"):
                has_words = "text" in conf_d and "conf" in conf_d
                if has_words:
                    words = conf_d["text"]
                    confs = conf_d["conf"]
                    rows  = [(w.strip(), c) for w, c in zip(words, confs) if w.strip() and c != -1]
                    low   = [(w, c) for w, c in rows if c < 50]
                    if low:
                        st.markdown(f'<div class="warn-box">⚠ {len(low)} word(s) below 50% confidence — check these manually</div>', unsafe_allow_html=True)
                        for w, c in sorted(low, key=lambda x: x[1])[:15]:
                            bar      = int(c / 5)
                            bar_html = f'<span style="color:#c62828">{"█"*bar}{"░"*(20-bar)} {c}%</span>'
                            st.markdown(
                                f'<div style="display:flex;gap:12px;padding:3px 0;font-family:monospace;font-size:0.76rem">'
                                f'<span style="min-width:120px;color:#1a1a2e">{safe_html(w)}</span>'
                                f'{bar_html}</div>', unsafe_allow_html=True
                            )
                    else:
                        st.markdown('<div class="info-box">✓ All detected words are above 50% confidence</div>', unsafe_allow_html=True)
                else:
                    confs = conf_d["conf"]
                    low_n = sum(1 for c in confs if 0 <= c < 50)
                    total = sum(1 for c in confs if c >= 0)
                    if low_n:
                        st.markdown(f'<div class="warn-box">⚠ {low_n}/{total} word regions below 50% confidence across all pages</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">✓ All detected regions are above 50% confidence</div>', unsafe_allow_html=True)
                    st.markdown('<div style="font-size:0.76rem;color:#90a4ae">Per-word breakdown is available for single images only.</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-icon">📭</span>
                <div class="empty-title">No text detected</div>
                <div class="empty-sub">Try a different binarization mode,<br>increase contrast, or enable Remove Shadows</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">📝</span>
            <div class="empty-title">No text extracted yet</div>
            <div class="empty-sub">Upload a file and click "Extract Text" to see the result</div>
        </div>""", unsafe_allow_html=True)
