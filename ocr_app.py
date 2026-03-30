"""
OCR Studio — Final Version
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
from pdf2image import convert_from_bytes

# ╔══════════════════════════════════════════════════════════════╗
# ║                    PATH CONFIGURATION                        ║
# ║  Edit these two lines to match your system                   ║
# ╚══════════════════════════════════════════════════════════════╝

# Tesseract executable — run `where tesseract` in CMD to find yours
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Poppler bin folder — download from:
# https://github.com/oschwartz10612/poppler-windows/releases
POPPLER_PATH = r"C:\poppler\Library\bin"

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
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --bg:          #0b0b0d;
    --bg-card:     #111113;
    --bg-surface:  #18181b;
    --bg-surface2: #1f1f23;
    --accent:      #e8ff47;
    --accent-dim:  rgba(232,255,71,0.10);
    --accent-glow: rgba(232,255,71,0.22);
    --green:       #4ade80;
    --green-dim:   rgba(74,222,128,0.10);
    --red:         #f87171;
    --red-dim:     rgba(248,113,113,0.10);
    --orange:      #fb923c;
    --text:        #ececec;
    --text-muted:  #777;
    --text-dim:    #444;
    --border:      rgba(255,255,255,0.06);
    --border-hi:   rgba(255,255,255,0.12);
    --border-acc:  rgba(232,255,71,0.28);
    --radius:      10px;
    --mono:        'Space Mono', monospace;
    --sans:        'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Upload ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-acc) !important;
    border-radius: var(--radius) !important;
    transition: all 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0b0b0d !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.18s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #f5ff7a !important;
    box-shadow: 0 0 18px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--border-acc) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
    width: 100%;
}
.stDownloadButton > button:hover { background: var(--accent-dim) !important; }

/* ── Form elements ── */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stRadio label, .stCheckbox label { color: var(--text) !important; }
hr { border-color: var(--border) !important; margin: 1.2rem 0; }
.stImage img { border-radius: var(--radius) !important; border: 1px solid var(--border) !important; }
[data-testid="column"] { padding: 0 0.4rem !important; }

/* ── Custom components ── */
.ocr-title {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 0.15rem;
}
.ocr-title span { color: var(--accent); }
.ocr-ver {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1.8rem;
}
.sec-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after { content:''; flex:1; height:1px; background:var(--border); }

.sb-sec {
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border);
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.9rem 0.8rem;
    text-align: center;
}
.stat-val {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.stat-lbl {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.25rem;
}

.result-box {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.3rem 1.4rem;
    font-family: var(--mono);
    font-size: 0.82rem;
    line-height: 1.85;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 420px;
    overflow-y: auto;
}
.result-box::-webkit-scrollbar { width: 5px; }
.result-box::-webkit-scrollbar-thumb { background: var(--border-acc); border-radius: 3px; }

.badge {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 0.64rem;
    padding: 0.18rem 0.55rem;
    border-radius: 20px;
    border: 1px solid var(--border-acc);
    letter-spacing: 0.5px;
}
.badge-pdf {
    display: inline-block;
    background: var(--red-dim);
    color: var(--red);
    font-family: var(--mono);
    font-size: 0.64rem;
    padding: 0.18rem 0.55rem;
    border-radius: 20px;
    border: 1px solid rgba(248,113,113,0.28);
}
.badge-ok {
    background: var(--green-dim);
    color: var(--green);
    border-color: rgba(74,222,128,0.28);
}

.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    background: var(--bg-card);
    color: var(--text-muted);
    font-size: 0.88rem;
}
.empty-icon { font-size: 2.6rem; opacity: 0.4; margin-bottom: 0.8rem; }

.page-tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent);
    background: var(--accent-dim);
    padding: 0.2rem 0.7rem;
    border-radius: 4px;
    margin: 0.8rem 0 0.3rem;
    letter-spacing: 1.5px;
}

.warn-box {
    background: rgba(251,146,60,0.08);
    border: 1px solid rgba(251,146,60,0.25);
    border-radius: var(--radius);
    padding: 0.7rem 0.9rem;
    font-size: 0.8rem;
    color: var(--orange);
    margin-bottom: 0.8rem;
}
.info-box {
    background: var(--accent-dim);
    border: 1px solid var(--border-acc);
    border-radius: var(--radius);
    padding: 0.7rem 0.9rem;
    font-size: 0.8rem;
    color: #c8d870;
    margin-bottom: 0.8rem;
}

/* heatmap legend */
.hm-legend { display:flex; gap:6px; align-items:center; font-size:0.72rem; color:var(--text-muted); margin-top:0.5rem; }
.hm-swatch { width:14px; height:14px; border-radius:3px; display:inline-block; }
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
    exe = os.path.join(POPPLER_PATH, "pdfinfo.exe")
    return os.path.isfile(exe)


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

    # ── Pillow adjustments ──
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)

    arr = np.array(img)

    # ── Upscale small images ──
    if upscale:
        arr = upscale_if_small(arr)

    # ── Shadow / uneven lighting removal ──
    if remove_shadow:
        arr = remove_shadows(arr)

    # ── Denoise ──
    if denoise:
        arr = cv2.fastNlMeansDenoisingColored(arr, None, h=10, hColor=10,
                                               templateWindowSize=7,
                                               searchWindowSize=21)

    # ── Deskew ──
    if deskew:
        arr = auto_deskew(arr)

    # ── Binarization mode ──
    if mode == "Grayscale":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if morph:
            gray = morph_clean(gray)
        arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    elif mode == "Otsu Binarize":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if morph:
            binary = morph_clean(binary)
        arr = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    elif mode == "Adaptive Threshold":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        if morph:
            adaptive = morph_clean(adaptive)
        arr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)

    elif mode == "CLAHE + Otsu":
        # Best for low-contrast images
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if morph:
            binary = morph_clean(binary)
        arr = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    elif mode == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
        arr = cv2.filter2D(arr, -1, kernel)

    elif mode == "Unsharp Mask":
        # Stronger sharpening that preserves edges
        blurred = cv2.GaussianBlur(arr, (0, 0), 3)
        arr = cv2.addWeighted(arr, 1.7, blurred, -0.7, 0)

    # ── White border padding — helps Tesseract detect edge text ──
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
    cfg += " -c tessedit_do_invert=0"          # don't try inverted image
    cfg += " -c textord_heavy_nr=1"            # aggressive noise removal
    cfg += " -c edges_max_children_per_outline=40"
    if preserve_spaces:
        cfg += " -c preserve_interword_spaces=1"
    if whitelist.strip():
        safe = re.sub(r'[^A-Za-z0-9 .,!?@#$%&*()\-_+=:;\'\"/<>]', '', whitelist)
        if safe:
            cfg += f" -c tessedit_char_whitelist={safe}"
    return cfg


def run_ocr(img: Image.Image, psm: int, oem: int,
            whitelist: str, preserve_spaces: bool,
            multi_run: bool) -> str:
    """
    Run Tesseract. If multi_run=True, runs with multiple PSM modes
    and returns the longest/best result (better for unknown layouts).
    """
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
        # pick result with most alphanumeric characters
        best = max(results, key=lambda t: len(re.findall(r'[A-Za-z0-9]', t))) if results else ""
        return best

    return pytesseract.image_to_string(img, lang="eng", config=cfg).strip()


def get_ocr_data(img: Image.Image, psm: int, oem: int,
                 whitelist: str, preserve_spaces: bool):
    cfg = build_config(psm, oem, whitelist, preserve_spaces)
    return pytesseract.image_to_data(
        img, lang="eng", config=cfg,
        output_type=pytesseract.Output.DICT
    )


# ═══════════════════════════════════════════════════════════════
#  POST-PROCESSING
# ═══════════════════════════════════════════════════════════════

# Common Tesseract character substitution mistakes
OCR_FIXES = [
    (r'(?<![A-Z0-9])\|(?![A-Z0-9])', 'I'),   # | → I (isolated)
    (r'(?<=\d)O(?=\d)', '0'),                  # digit-O-digit → 0
    (r'(?<=\d)l(?=\d)', '1'),                  # digit-l-digit → 1
    (r'(?<=\d)I(?=\d)', '1'),                  # digit-I-digit → 1
    (r'(?<=[A-Za-z])0(?=[A-Za-z])', 'O'),      # letter-0-letter → O
    (r'\brn\b', 'm'),                           # rn → m (common split)
    (r'(?<!\w)l\.', 'I.'),                      # l. at word start → I.
    (r'\s{2,}', ' '),                           # collapse multiple spaces
    (r'[ \t]+\n', '\n'),                        # trailing spaces on lines
    (r'\n{3,}', '\n\n'),                        # max 2 consecutive blank lines
]


def post_process(text: str, fix_chars: bool, remove_junk: bool) -> str:
    """Clean up raw Tesseract output."""
    if not text:
        return text

    # Normalize unicode (e.g. smart quotes → ASCII)
    text = unicodedata.normalize("NFKC", text)

    if fix_chars:
        for pattern, replacement in OCR_FIXES:
            text = re.sub(pattern, replacement, text)

    if remove_junk:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            # drop lines with fewer than 2 real alphanumeric characters
            alnum = re.sub(r'[^a-zA-Z0-9]', '', line)
            if len(alnum) >= 2:
                cleaned.append(line)
            elif line.strip() == "":
                cleaned.append("")   # preserve paragraph breaks
        text = "\n".join(cleaned)
        # final blank-line collapse
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

        # Color: green (high) → orange (mid) → red (low)
        if conf >= 80:
            color = (74, 222, 128)    # green
            alpha = 0.25
        elif conf >= 50:
            color = (251, 191, 36)    # amber
            alpha = 0.35
        else:
            color = (248, 113, 113)   # red
            alpha = 0.45

        overlay = arr.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        arr = cv2.addWeighted(overlay, alpha, arr, 1 - alpha, 0)
        cv2.rectangle(arr, (x, y), (x + w, y + h), color, 1)

    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════════
#  STATS HELPERS
# ═══════════════════════════════════════════════════════════════

def word_count(text):
    return len(text.split()) if text else 0

def char_count(text):
    return len(re.sub(r'\s', '', text)) if text else 0

def line_count(text):
    return len([l for l in text.split('\n') if l.strip()]) if text else 0

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
    st.markdown('<div class="ocr-title">OCR<span>.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="ocr-ver">Studio · Final Edition</div>', unsafe_allow_html=True)

    # ── Status ──
    st.markdown('<div class="sb-sec">System</div>', unsafe_allow_html=True)
    t_badge = f'<span class="badge badge-ok">✓ Tesseract {tess_ver}</span>' if tess_ok else '<span class="badge" style="color:var(--red);border-color:rgba(248,113,113,.3)">✗ Not found</span>'
    p_badge = '<span class="badge badge-ok">✓ Poppler ready</span>' if popp_ok else '<span class="badge" style="color:var(--orange);border-color:rgba(251,146,60,.3)">✗ PDF disabled</span>'
    st.markdown(f'<div style="display:flex;flex-direction:column;gap:5px;margin-bottom:0.5rem">{t_badge}{p_badge}</div>', unsafe_allow_html=True)
    if not tess_ok:
        st.error("Fix `tesseract_cmd` path at top of ocr_app.py")
    if not popp_ok:
        st.warning("Fix `POPPLER_PATH` at top of ocr_app.py for PDF support")

    # ── Preprocessing ──
    st.markdown('<div class="sb-sec">Preprocessing</div>', unsafe_allow_html=True)

    mode = st.selectbox("Binarization mode", [
        "Adaptive Threshold",
        "CLAHE + Otsu",
        "Otsu Binarize",
        "Grayscale",
        "Unsharp Mask",
        "Sharpen",
        "Original",
    ], index=0, help="Adaptive Threshold and CLAHE+Otsu give best accuracy on most documents")

    col1, col2 = st.columns(2)
    with col1:
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
        contrast   = st.slider("Contrast",   0.5, 3.0, 1.3, 0.05)
    with col2:
        sharpness  = st.slider("Sharpness",  0.5, 3.0, 1.2, 0.05)

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        denoise       = st.checkbox("Denoise",        value=True,  help="Remove camera grain")
        deskew        = st.checkbox("Deskew",         value=True,  help="Fix tilted scans")
        remove_shadow = st.checkbox("Remove shadows", value=True,  help="Normalize lighting")
    with c2:
        upscale       = st.checkbox("Upscale",        value=True,  help="Scale up small images")
        morph         = st.checkbox("Morph clean",    value=True,  help="Fill character gaps")
        border_pad    = st.checkbox("Border pad",     value=True,  help="Add white padding")

    # ── Tesseract Config ──
    st.markdown('<div class="sb-sec">Tesseract Engine</div>', unsafe_allow_html=True)

    psm_map = {
        "3 — Fully automatic":           3,
        "4 — Single column":             4,
        "6 — Uniform text block":        6,
        "7 — Single text line":          7,
        "8 — Single word":               8,
        "11 — Sparse text":              11,
        "12 — Sparse + OSD":             12,
        "13 — Raw line":                 13,
    }
    psm_label = st.selectbox("Page segmentation (PSM)", list(psm_map.keys()), index=0)
    psm = psm_map[psm_label]

    oem_map = {
        "3 — LSTM + Legacy (best)": 3,
        "1 — LSTM only":            1,
        "0 — Legacy only":          0,
    }
    oem_label = st.selectbox("OCR engine (OEM)", list(oem_map.keys()), index=0)
    oem = oem_map[oem_label]

    preserve_spaces = st.checkbox("Preserve spacing", value=True)
    multi_run = st.checkbox("Multi-PSM mode", value=False,
                            help="Run multiple PSM modes and pick best result (slower but more accurate for unknown layouts)")

    whitelist = st.text_input("Character whitelist", value="",
                              placeholder="e.g. 0123456789ABCDEFabcdef",
                              help="Leave empty to allow all characters. Use for IDs, numbers, codes.")

    # ── Post-processing ──
    st.markdown('<div class="sb-sec">Post-processing</div>', unsafe_allow_html=True)
    fix_chars    = st.checkbox("Fix common OCR errors", value=True,
                               help="Auto-correct |→I, 0↔O, l↔1 in context")
    remove_junk  = st.checkbox("Remove junk lines",     value=True,
                               help="Drop lines with < 2 real characters")
    show_heatmap = st.checkbox("Confidence heatmap",    value=True,
                               help="Visualize per-word confidence on image")

    # ── PDF Settings ──
    if popp_ok:
        st.markdown('<div class="sb-sec">PDF Settings</div>', unsafe_allow_html=True)
        pdf_dpi = st.slider("Render DPI", 200, 500, 350, 50,
                            help="Higher = better quality but slower. 350 is optimal for most PDFs.")
    else:
        pdf_dpi = 350

    st.markdown('<div class="sb-sec">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.74rem;color:#555;line-height:1.8">
    Engine: <b style="color:#666">Tesseract 5 LSTM</b><br>
    Vision: <b style="color:#666">OpenCV 4</b><br>
    Formats: JPG · PNG · BMP · TIFF · PDF<br>
    Language: English
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN — LEFT COLUMN (INPUT)
# ═══════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="sec-label">Input</div>', unsafe_allow_html=True)

    allowed = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
    if popp_ok:
        allowed.append("pdf")

    uploaded = st.file_uploader(
        "Drop image or PDF" if popp_ok else "Drop image (PDF requires Poppler)",
        type=allowed,
        label_visibility="collapsed"
    )

    is_pdf = uploaded is not None and uploaded.name.lower().endswith(".pdf")

    # ── PDF handling ──
    if uploaded and is_pdf:
        st.markdown(f"""
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:0.8rem">
            <span class="badge-pdf">PDF</span>
            <span class="badge">{uploaded.name}</span>
            <span class="badge">{round(uploaded.size/1024,1)} KB</span>
        </div>""", unsafe_allow_html=True)

        with st.spinner("Converting PDF → images..."):
            try:
                raw = uploaded.read()
                pdf_pages = convert_from_bytes(raw, dpi=pdf_dpi, poppler_path=POPPLER_PATH)
                st.session_state["pdf_pages"] = pdf_pages
                st.success(f"{len(pdf_pages)} page(s) loaded at {pdf_dpi} DPI")
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
        <div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:0.7rem">
            <span class="badge">{w}×{h}px</span>
            <span class="badge">→ {pw}×{ph}px</span>
            <span class="badge">{uploaded.name}</span>
            <span class="badge">{round(uploaded.size/1024,1)} KB</span>
            <span class="badge">{mode}</span>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            Upload an image or PDF to begin<br>
            <span style="font-size:0.78rem;opacity:0.5">JPG · PNG · BMP · TIFF · PDF</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded and tess_ok:
        run_btn = st.button("⚡  Extract Text", use_container_width=True)
    elif uploaded and not tess_ok:
        st.markdown('<div class="warn-box">Tesseract not found — set the path at the top of ocr_app.py</div>', unsafe_allow_html=True)
        run_btn = False
    else:
        run_btn = False


# ═══════════════════════════════════════════════════════════════
#  MAIN — RIGHT COLUMN (OUTPUT)
# ═══════════════════════════════════════════════════════════════

with col_right:
    st.markdown('<div class="sec-label">Output</div>', unsafe_allow_html=True)

    # ── Run OCR ──
    if uploaded and run_btn and tess_ok:
        start = time.time()

        if is_pdf and st.session_state.get("pdf_pages"):
            pages = st.session_state["pdf_pages"]
            all_text, all_confs, all_data_last = [], [], None
            prog = st.progress(0, text="Extracting text from pages...")

            for i, page in enumerate(pages):
                proc = preprocess_image(
                    page, mode, brightness, contrast, sharpness,
                    denoise, deskew, remove_shadow, upscale, morph, border_pad
                )
                raw_text = run_ocr(proc, psm, oem, whitelist, preserve_spaces, multi_run)
                clean    = post_process(raw_text, fix_chars, remove_junk)
                pdata    = get_ocr_data(proc, psm, oem, whitelist, preserve_spaces)
                all_text.append(f"[PAGE {i+1}]\n{clean}")
                all_confs.extend([c for c in pdata["conf"] if c != -1])
                all_data_last = pdata
                prog.progress((i + 1) / len(pages), text=f"Page {i+1}/{len(pages)}...")

            prog.empty()
            extracted_text   = "\n\n".join(all_text)
            combined_conf    = {"conf": all_confs}
            elapsed          = round(time.time() - start, 2)
            pages_n          = len(pages)
            heatmap_img      = draw_confidence_heatmap(pages[0], all_data_last) if show_heatmap and all_data_last else None

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
        conf_col = "#4ade80" if conf >= 80 else "#fbbf24" if conf >= 55 else "#f87171"

        # Stats
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="stat-card"><div class="stat-val">{word_count(text)}</div><div class="stat-lbl">Words</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-val">{char_count(text)}</div><div class="stat-lbl">Chars</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-val">{line_count(text)}</div><div class="stat-lbl">Lines</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{conf_col}">{conf}%</div><div class="stat-lbl">Confidence</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="stat-card"><div class="stat-val">{"{}p".format(pages_n) if pages_n>1 else "{}s".format(elapsed)}</div><div class="stat-lbl">{"Pages" if pages_n>1 else "Time"}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if text:
            # ── Result text ──
            st.markdown(f'<div class="result-box">{safe_html(text)}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Downloads ──
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button("⬇ Text (.txt)", data=text.encode("utf-8"),
                                   file_name="ocr_output.txt", mime="text/plain",
                                   use_container_width=True)
            with d2:
                # Markdown-friendly version
                md = text.replace("\n", "  \n")
                st.download_button("⬇ Markdown (.md)", data=md.encode("utf-8"),
                                   file_name="ocr_output.md", mime="text/markdown",
                                   use_container_width=True)
            with d3:
                if is_pdf and st.session_state.get("pdf_pages"):
                    src = preprocess_image(
                        st.session_state["pdf_pages"][0],
                        mode, brightness, contrast, sharpness,
                        denoise, deskew, remove_shadow, upscale, morph, border_pad
                    )
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
                        <span class="hm-swatch" style="background:#4ade80"></span> ≥80% confident
                        &nbsp;
                        <span class="hm-swatch" style="background:#fbbf24"></span> 50–79%
                        &nbsp;
                        <span class="hm-swatch" style="background:#f87171"></span> &lt;50% — review these
                    </div>""", unsafe_allow_html=True)
                    hm_bytes = image_to_bytes(heat_img)
                    st.download_button("⬇ Download heatmap", data=hm_bytes,
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
                            f'<div style="display:flex;gap:10px;padding:5px 0;border-bottom:1px solid #18181b">'
                            f'<span style="font-family:var(--mono,monospace);font-size:0.65rem;color:#444;min-width:22px">{n:02d}</span>'
                            f'<span style="font-size:0.82rem;color:#ccc">{safe_html(line)}</span>'
                            f'</div>', unsafe_allow_html=True
                        )

            # ── Word confidence table ──
            with st.expander("📊 Word confidence table"):
                # PDF mode stores only {"conf": [...]} — no "text" key
                has_words = "text" in conf_d and "conf" in conf_d
                if has_words:
                    words = conf_d["text"]
                    confs = conf_d["conf"]
                    rows  = [(w.strip(), c) for w, c in zip(words, confs)
                             if w.strip() and c != -1]
                    low   = [(w, c) for w, c in rows if c < 50]
                    if low:
                        st.markdown(f'<div class="warn-box">⚠ {len(low)} word(s) below 50% confidence — check these manually</div>', unsafe_allow_html=True)
                        for w, c in sorted(low, key=lambda x: x[1])[:15]:
                            bar      = int(c / 5)
                            bar_html = f'<span style="color:#f87171">{"█"*bar}{"░"*(20-bar)} {c}%</span>'
                            st.markdown(
                                f'<div style="display:flex;gap:12px;padding:3px 0;font-family:monospace;font-size:0.78rem">'
                                f'<span style="min-width:120px;color:#eee">{safe_html(w)}</span>'
                                f'{bar_html}'
                                f'</div>', unsafe_allow_html=True
                            )
                    else:
                        st.markdown('<div class="info-box">✓ All detected words are above 50% confidence</div>', unsafe_allow_html=True)
                else:
                    # PDF multi-page: only aggregate conf scores available
                    confs = conf_d["conf"]
                    low_n = sum(1 for c in confs if 0 <= c < 50)
                    total = sum(1 for c in confs if c >= 0)
                    if low_n:
                        st.markdown(f'<div class="warn-box">⚠ {low_n}/{total} word regions below 50% confidence across all pages</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">✓ All detected regions are above 50% confidence</div>', unsafe_allow_html=True)
                    st.markdown('<div style="font-size:0.78rem;color:var(--text-muted)">Per-word breakdown is available for single images only.</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">📭</div>
                No text detected.<br>
                <span style="font-size:0.78rem;opacity:0.5">
                Try: higher DPI · different binarization mode · enable Remove shadows
                </span>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📝</div>
            Results will appear here<br>
            <span style="font-size:0.78rem;opacity:0.5">Upload a file and click Extract Text</span>
        </div>""", unsafe_allow_html=True)
