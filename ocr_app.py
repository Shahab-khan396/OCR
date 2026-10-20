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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CSS — Clean light theme with teal accents
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:           #f5f6fa;
    --bg-white:     #ffffff;
    --panel-bg:     #ffffff;
    --accent:       #00897B;
    --accent-light: rgba(0,137,123,0.08);
    --accent-mid:   rgba(0,137,123,0.18);
    --accent-dark:  #00695C;
    --teal-glow:    rgba(0,137,123,0.22);
    --green:        #2e7d32;
    --green-bg:     #e8f5e9;
    --red:          #c62828;
    --red-bg:       #ffebee;
    --orange:       #e65100;
    --orange-bg:    #fff3e0;
    --text:         #1a1a2e;
    --text-sub:     #546e7a;
    --text-muted:   #90a4ae;
    --border:       #e0e7ef;
    --border-hi:    #c8d6e0;
    --radius:       12px;
    --radius-sm:    8px;
    --shadow-sm:    0 1px 4px rgba(0,0,0,0.07);
    --sans:         'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-white) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 0 !important; padding-left: 0 !important; padding-right: 0 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stFileUploader"] label { display: none !important; }

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; border-color: var(--accent) !important; }

/* ── Selectboxes ── */
[data-testid="stSelectbox"] > div > div { background: var(--bg) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; color: var(--text) !important; }

/* ── Text input ── */
[data-testid="stTextInput"] > div > div > input { background: var(--bg) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; color: var(--text) !important; }

/* ── Images ── */
.stImage img { border-radius: var(--radius) !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow-sm) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] { background: var(--bg) !important; border-radius: var(--radius-sm) !important; padding: 3px !important; gap: 2px !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { background: transparent !important; border-radius: 6px !important; color: var(--text-sub) !important; font-size: 0.8rem !important; }
[data-testid="stTabs"] [aria-selected="true"] { background: var(--bg-white) !important; color: var(--accent) !important; font-weight: 600 !important; box-shadow: var(--shadow-sm) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; background: var(--bg-white) !important; }

/* ── Columns ── */
[data-testid="column"] { padding: 0 0.5rem !important; }

/* ════════════════════════════════════════
   SIDEBAR COMPONENTS
════════════════════════════════════════ */
.sb-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 16px 16px 12px; border-bottom: 1px solid var(--border);
}
.sb-brand-icon {
    width: 34px; height: 34px; background: var(--accent);
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 16px; color: white; flex-shrink: 0;
}
.sb-brand-name { font-size: 0.95rem; font-weight: 700; color: var(--text); letter-spacing: -0.2px; display: block; }
.sb-brand-sub  { font-size: 0.67rem; color: var(--text-muted); display: block; margin-top: 1px; }

.sb-nav { padding: 6px 8px 0; }
.sb-nav-item {
    display: flex; align-items: center; gap: 9px; padding: 7px 10px;
    border-radius: var(--radius-sm); font-size: 0.83rem; font-weight: 500;
    color: var(--text-sub); margin-bottom: 2px;
}
.sb-nav-item.active { background: var(--accent-light); color: var(--accent); font-weight: 600; }
.sb-nav-icon { font-size: 14px; width: 17px; text-align: center; }

.sb-section-label {
    font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--text-muted); padding: 12px 16px 5px;
}
.sb-subsection {
    font-size: 0.72rem; font-weight: 600; color: var(--text-sub);
    padding: 8px 16px 4px; display: flex; align-items: center; gap: 6px;
}
.sb-subsection-icon { font-size: 12px; color: var(--accent); }
.sb-divider { height: 1px; background: var(--border); margin: 6px 0; }

.status-row { display: flex; gap: 6px; flex-wrap: wrap; padding: 0 16px 10px; }
.status-pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 20px; font-size: 0.68rem; font-weight: 600;
}
.status-ok   { background: var(--green-bg); color: var(--green); }
.status-warn { background: var(--orange-bg); color: var(--orange); }
.status-err  { background: var(--red-bg); color: var(--red); }

/* ════════════════════════════════════════
   PANEL CARDS
════════════════════════════════════════ */
.panel-card { background: var(--panel-bg); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow-sm); overflow: hidden; }
.panel-header { display: flex; align-items: center; justify-content: space-between; padding: 13px 16px 11px; border-bottom: 1px solid var(--border); }
.panel-header-left { display: flex; align-items: center; gap: 9px; }
.panel-icon { width: 28px; height: 28px; background: var(--accent-light); border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 13px; color: var(--accent); flex-shrink: 0; }
.panel-title    { font-size: 0.88rem; font-weight: 700; color: var(--text); line-height: 1; }
.panel-subtitle { font-size: 0.68rem; color: var(--text-muted); margin-top: 1px; }
.panel-badges   { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }
.fmt-badge { font-size: 0.6rem; font-weight: 600; color: var(--text-muted); background: var(--bg); border: 1px solid var(--border); padding: 1px 5px; border-radius: 4px; letter-spacing: 0.3px; }
.panel-body { padding: 16px; }

/* ── Drop zone ── */
.drop-zone {
    border: 2px dashed var(--accent-mid); border-radius: var(--radius);
    padding: 30px 20px; text-align: center; background: var(--accent-light); margin-bottom: 4px;
}
.drop-icon  { font-size: 2.2rem; color: var(--accent); margin-bottom: 9px; display: block; }
.drop-title { font-size: 0.88rem; font-weight: 600; color: var(--text); margin-bottom: 3px; }
.drop-sub   { font-size: 0.75rem; color: var(--text-muted); }

/* ── Pro tips ── */
.pro-tips { background: #fffde7; border: 1px solid #f9a825; border-radius: var(--radius-sm); padding: 10px 13px; margin-top: 12px; }
.pro-tips-title { font-size: 0.76rem; font-weight: 700; color: #f57f17; margin-bottom: 5px; display: flex; align-items: center; gap: 5px; }
.pro-tips ul { margin: 0; padding-left: 15px; font-size: 0.73rem; color: #5d4037; line-height: 1.7; }

/* ── Stats ── */
.stat-row { display: flex; gap: 7px; margin-bottom: 12px; }
.stat-card { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 9px 6px; text-align: center; }
.stat-val { font-size: 1.25rem; font-weight: 700; color: var(--accent); line-height: 1; font-variant-numeric: tabular-nums; }
.stat-lbl { font-size: 0.58rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.7px; margin-top: 3px; font-weight: 600; }

/* ── Result box ── */
.result-box {
    background: var(--bg); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: var(--radius-sm);
    padding: 13px 15px; font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem; line-height: 1.8; color: var(--text);
    white-space: pre-wrap; word-break: break-word;
    max-height: 380px; overflow-y: auto;
}
.result-box::-webkit-scrollbar { width: 4px; }
.result-box::-webkit-scrollbar-thumb { background: var(--accent-mid); border-radius: 2px; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 50px 24px; color: var(--text-muted); }
.empty-icon  { font-size: 3rem; opacity: 0.28; margin-bottom: 14px; display: block; }
.empty-title { font-size: 0.88rem; font-weight: 600; color: var(--text-sub); margin-bottom: 5px; }
.empty-sub   { font-size: 0.76rem; color: var(--text-muted); line-height: 1.5; }

/* ── Page tag ── */
.page-tag { display: inline-block; font-size: 0.62rem; font-weight: 700; color: var(--accent); background: var(--accent-light); padding: 2px 7px; border-radius: 4px; margin: 9px 0 3px; letter-spacing: 1px; text-transform: uppercase; }

/* ── Alert boxes ── */
.warn-box { background: var(--orange-bg); border: 1px solid rgba(230,81,0,0.22); border-radius: var(--radius-sm); padding: 8px 11px; font-size: 0.78rem; color: var(--orange); margin-bottom: 9px; }
.info-box { background: var(--green-bg); border: 1px solid rgba(46,125,50,0.22); border-radius: var(--radius-sm); padding: 8px 11px; font-size: 0.78rem; color: var(--green); margin-bottom: 9px; }

/* ── Heatmap ── */
.hm-legend { display:flex; gap:7px; align-items:center; font-size:0.7rem; color:var(--text-sub); margin-top:5px; }
.hm-swatch { width:11px; height:11px; border-radius:3px; display:inline-block; flex-shrink:0; }

/* ── Info badges ── */
.info-badge { display: inline-block; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; font-size: 0.66rem; font-weight: 500; color: var(--text-sub); padding: 2px 7px; margin: 2px 2px 0 0; }
.info-badge.accent { background: var(--accent-light); border-color: var(--accent-mid); color: var(--accent); font-weight: 600; }
.badge { display: inline-block; background: var(--accent-light); border: 1px solid var(--accent-mid); border-radius: 4px; font-size: 0.66rem; font-weight: 500; color: var(--accent); padding: 2px 7px; margin: 2px 2px 0 0; }
.badge-pdf { display: inline-block; background: var(--red-bg); border: 1px solid rgba(198,40,40,0.2); border-radius: 4px; font-size: 0.66rem; font-weight: 600; color: var(--red); padding: 2px 7px; margin: 2px 2px 0 0; }
.badge-ok  { background: var(--green-bg); border: 1px solid rgba(46,125,50,0.2); color: var(--green); }

/* ── Download buttons ── */
.stDownloadButton > button { background: var(--bg) !important; color: var(--accent) !important; border: 1px solid var(--accent-mid) !important; border-radius: var(--radius-sm) !important; font-size: 0.76rem !important; font-weight: 600 !important; transition: all 0.15s !important; }
.stDownloadButton > button:hover { background: var(--accent-light) !important; border-color: var(--accent) !important; }

/* ── Primary button ── */
.stButton > button { background: var(--accent) !important; color: #ffffff !important; font-weight: 700 !important; font-size: 0.86rem !important; border: none !important; border-radius: var(--radius-sm) !important; padding: 0.65rem 1.4rem !important; transition: all 0.18s !important; box-shadow: 0 2px 8px var(--teal-glow) !important; letter-spacing: 0.2px !important; }
.stButton > button:hover { background: var(--accent-dark) !important; box-shadow: 0 4px 14px var(--teal-glow) !important; transform: translateY(-1px) !important; }
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
