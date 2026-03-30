|<p>**OCR Studio**</p><p>*Final Edition  ·  Maximum Accuracy OCR Application*</p><p>Python  ·  Streamlit  ·  Tesseract 5  ·  OpenCV 4  ·  Windows</p>|
| :- |

**OVERVIEW**

OCR Studio is a professional-grade Optical Character Recognition application built with Streamlit. It extracts text from images and PDF documents using a multi-stage preprocessing pipeline powered by Tesseract 5 (LSTM engine) and OpenCV 4. The application is designed for maximum accuracy on real-world documents — scanned pages, photographed text, invoices, forms, and PDFs with uneven lighting or skew.

Every stage of the pipeline is configurable from the sidebar: binarization mode, brightness, contrast, sharpness, deskewing, shadow removal, upscaling, morphological cleanup, and more. Post-processing corrects common Tesseract character errors in context and removes junk lines. A confidence heatmap overlays colour-coded bounding boxes on the source image so you can see exactly which words were read with low certainty.

**QUICK START**

**1.  Install Python dependencies**

pip install -r requirements.txt

**2.  Install Tesseract (required)**

Tesseract is the OCR engine. pytesseract is only a Python wrapper — the binary must be installed separately.

- Download the Windows installer from:  https://github.com/UB-Mannheim/tesseract/wiki
- Run the installer. Default path:  C:\Program Files\Tesseract-OCR\
- Verify installation by running in Command Prompt:

tesseract --version

**3.  Install Poppler  (required for PDF support)**

Poppler converts PDF pages to images before OCR. Without it, PDF uploads are disabled.

- Download the Windows release from:  https://github.com/oschwartz10612/poppler-windows/releases
- Extract the zip to a permanent location, e.g.  C:\poppler\
- The bin folder path should look like:  C:\poppler\Library\bin
- Confirm by running in Command Prompt:

"C:\poppler\Library\bin\pdfinfo.exe" --version

**4.  Configure paths in ocr\_app.py**

Open ocr\_app.py and update the two lines at the top of the file:

pytesseract.pytesseract.tesseract\_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPPLER\_PATH = r"C:\poppler\Library\bin"

|●  If you installed to a different directory, run  where tesseract  in Command Prompt to find the exact path, and update accordingly.|
| :- |

**5.  Launch the application**

streamlit run ocr\_app.py

The app opens automatically in your default browser at  http://localhost:8501

**REQUIREMENTS**

**Python packages  (requirements.txt)**

|**Package**|**Version**|**Purpose**|
| :- | :- | :- |
|streamlit|≥ 1.35.0|Web application framework|
|pytesseract|0\.3.13|Python wrapper for Tesseract OCR|
|opencv-python|4\.13.0.92|Image processing and computer vision|
|Pillow|12\.1.1|Image loading, enhancement, export|
|numpy|2\.4.2|Array operations for pixel manipulation|
|pdf2image|1\.17.0|PDF → image conversion via Poppler|

**System requirements**

- Operating System:  Windows 10 / 11  (64-bit)
- Python:  3.9 or later
- Tesseract OCR:  version 5.x  (LSTM engine required)
- Poppler:  latest Windows release  (PDF support only)
- RAM:  minimum 4 GB recommended;  8 GB for large PDFs at high DPI

**FEATURES**

**Preprocessing pipeline**

All preprocessing steps run before Tesseract. Each can be toggled independently from the sidebar. The recommended combination for most documents is: Adaptive Threshold + Denoise + Deskew + Remove Shadows + Upscale + Morph Clean + Border Pad.

**Binarization modes (6)**  Adaptive Threshold — best for uneven lighting; CLAHE + Otsu — best for low contrast; Otsu Binarize — fast and effective on clean scans; Grayscale; Unsharp Mask; Sharpen; Original

**Brightness / Contrast / Sharpness**  Pillow ImageEnhance sliders (0.5–3.0 range). Applied before all OpenCV operations.

**Denoise**  OpenCV fastNlMeansDenoisingColored — removes JPEG compression artefacts and camera grain.

**Auto-deskew**  Hough line transform detects the median text angle and corrects rotation automatically. Skips correction when tilt is under 0.3°.

**Shadow removal**  Morphological dilation + median blur normalization removes dark gradients from page edges, folder shadows, and uneven scan lighting.

**Upscaling**  Lanczos4 interpolation scales images narrower than 2000px up to 2000px width. Tesseract accuracy degrades sharply below ~150 DPI.

**Morphological cleanup**  Morphological CLOSE fills broken character strokes; OPEN removes isolated noise dots. Applied after binarization.

**White border padding**  Adds 30px white border around the image so Tesseract does not miss characters at the edges of the frame.

**Tesseract engine configuration**

- PSM (Page Segmentation Mode) — 8 options: auto, single column, block, line, word, sparse, raw
- OEM (OCR Engine Mode) — LSTM + Legacy, LSTM only, Legacy only
- Multi-PSM mode — runs PSM 3, 6, 11, and selected mode in parallel; picks the result with the most alphanumeric content
- Character whitelist — restrict recognition to specific characters (e.g. digits only for invoice amounts)
- Preserve interword spacing — maintains original word spacing in output
- Extra flags: tessedit\_do\_invert=0, textord\_heavy\_nr=1 for improved noise handling

**Post-processing**

- Unicode normalization (NFKC) — converts smart quotes, ligatures, and non-standard characters to standard ASCII equivalents
- Context-aware character fixes — corrects | → I, digit-O-digit → 0, digit-l-digit → 1, rn → m only in the correct context
- Junk line removal — strips lines containing fewer than 2 alphanumeric characters; preserves paragraph breaks
- Whitespace normalization — collapses multiple spaces and excessive blank lines

**PDF support**

- Multi-page PDF processing via pdf2image + Poppler
- Configurable render DPI (200–500; default 350)
- Page-by-page progress bar during extraction
- Output tagged with [PAGE n] markers for each page
- Preview any page before extracting with the page slider

**Output and export**

- Extracted text displayed in a scrollable monospace result box
- Download as plain text (.txt) or Markdown (.md)
- Download the processed image (.png) showing all applied transformations
- Confidence heatmap — colour-coded bounding boxes (green ≥80%, amber 50–79%, red <50%) overlaid on the image
- Word confidence table — lists every word below 50% confidence with a visual bar; PDF mode shows aggregate summary
- Line-by-line breakdown — expandable section showing each line with its line number
- Stats row — word count, character count, line count, average confidence %, processing time

**SETTINGS REFERENCE**

**Recommended settings by document type**

|**Document type**|**Binarization**|**Key toggles**|**PSM**|
| :- | :- | :- | :- |
|Printed document / book scan|Adaptive Threshold|Denoise, Deskew, Upscale, Morph|PSM 6|
|Phone photo of text|CLAHE + Otsu|Remove Shadows, Denoise, Upscale, Deskew|PSM 3|
|Invoice / receipt|Otsu Binarize|Upscale, Morph, Border Pad|PSM 6|
|Single line / caption|Adaptive Threshold|Upscale, Morph|PSM 7|
|ID card / serial number|CLAHE + Otsu|Upscale, Morph, Whitelist: digits+letters|PSM 7 or 8|
|Low contrast faded text|CLAHE + Otsu|Contrast boost, Denoise, Morph|PSM 6|
|Handwritten notes|Adaptive Threshold|Remove Shadows, Denoise, Deskew|PSM 6|
|Multi-column newspaper / PDF|Adaptive Threshold|Deskew, Upscale, Morph, Multi-PSM|PSM 4|

**PSM mode guide**

**PSM 3**  Fully automatic page segmentation. Best default for unknown layouts.

**PSM 4**  Single column of text. Good for narrow documents and PDFs with one column.

**PSM 6**  Uniform block of text. Best for well-formatted printed documents.

**PSM 7**  Single text line. Use for captions, headings, single-line inputs.

**PSM 8**  Single word. Use for highly cropped images with one word.

**PSM 11**  Sparse text. Finds as much text as possible regardless of layout.

**PSM 12**  Sparse text with OSD. PSM 11 with orientation detection.

**PSM 13**  Raw line. Treats image as a single raw text line, bypassing hacks.

**TROUBLESHOOTING**

**TesseractNotFoundError**

*Tesseract binary is not installed or the path in ocr\_app.py is wrong.*

- Run  where tesseract  in Command Prompt to find the exact path.
- Open ocr\_app.py and update the  tesseract\_cmd  variable at the top.
- Alternatively, add the Tesseract directory to your system PATH environment variable.

**PDFInfoNotInstalledError**

*Poppler is not installed or POPPLER\_PATH points to the wrong folder.*

- Download from  https://github.com/oschwartz10612/poppler-windows/releases
- Extract and note the full path to the  bin  folder (contains pdfinfo.exe).
- Verify:  "C:\poppler\Library\bin\pdfinfo.exe" --version
- Update  POPPLER\_PATH  in ocr\_app.py to match.

**Low confidence score (< 50%)**

*The image quality is insufficient for reliable text recognition.*

- Enable Remove Shadows — fixes dark edges from scanning.
- Switch binarization to CLAHE + Otsu for low-contrast images.
- Increase Contrast slider to 1.5–2.0.
- Enable Upscale to ensure the image is at least 2000px wide.
- For PDFs, increase render DPI to 400 or 450.
- Enable Multi-PSM mode to try multiple layout interpretations.

**No text detected**

*Tesseract could not find any text regions in the processed image.*

- Try a different binarization mode — Adaptive Threshold works on most images.
- Disable Denoise if the image is already clean (it can blur thin text).
- Disable Border Pad and try again.
- Try PSM 11 (sparse text) for images where text is scattered across the page.
- Check the Processed tab — if the image is all black or all white, adjust brightness/contrast.

**Characters misread (O vs 0, l vs 1, | vs I)**

*Common Tesseract substitution errors on low-resolution or stylised fonts.*

- Enable Fix common OCR errors in Post-processing (on by default).
- Use the Character whitelist to restrict recognition (e.g. digits only).
- For numeric data: whitelist  0123456789.,  to eliminate letter confusion.
- Increase DPI (for PDFs) or enable Upscale (for images).

**Skewed or rotated output**

*The source image is tilted and Deskew did not fully correct it.*

- Enable Deskew in the sidebar (on by default).
- For severely rotated images (> 45°), pre-rotate the image before uploading.
- Enable Remove Shadows — shadows on a white page can confuse the angle detector.

**PROJECT STRUCTURE**

OCR/

├── ocr\_app.py          Main application — all logic in one file

├── requirements.txt    Python package dependencies

└── README.docx         This document

The entire application is self-contained in ocr\_app.py. There are no separate modules, config files, or asset folders required. The only external dependencies are the Tesseract binary and (optionally) Poppler.

**HOW IT WORKS**

The application processes each file through five sequential stages:

**1.  Upload**  Streamlit reads the uploaded file into memory. Images are opened with Pillow. PDFs are converted page-by-page to PIL Image objects using pdf2image and Poppler at the configured DPI.

**2.  Preprocessing**  The image is passed through the configurable OpenCV and Pillow pipeline: brightness/contrast/sharpness adjustment → upscaling → shadow removal → denoising → deskewing → binarization mode → morphological cleanup → border padding. Each step is applied only if its sidebar toggle is enabled.

**3.  OCR (Tesseract 5)**  The preprocessed image is passed to Tesseract via pytesseract. In standard mode, a single run with the selected PSM and OEM is performed. In Multi-PSM mode, four runs are performed in parallel and the result with the highest alphanumeric character count is selected. image\_to\_data() is called separately to obtain per-word confidence scores and bounding box coordinates.

**4.  Post-processing**  The raw text is normalized with unicodedata.normalize(NFKC), then context-aware regex substitutions fix common character errors. Junk lines (fewer than 2 alphanumeric characters) are removed if enabled. Multiple spaces and excessive blank lines are collapsed.

**5.  Output**  The cleaned text is displayed in a scrollable box. Stats (word count, character count, line count, confidence, time) are computed from the text and Tesseract data. The confidence heatmap draws colour-coded rectangles using OpenCV onto the processed image. Downloads are served as in-memory byte buffers — nothing is written to disk.

**KNOWN LIMITATIONS**

- Language support — only English is configured. Additional languages require installing the corresponding Tesseract language pack (e.g. tesseract-ocr-urd for Urdu) and updating the lang parameter in run\_ocr().
- Handwritten text — accuracy on cursive or irregular handwriting is limited with Tesseract. Consider EasyOCR or a vision-language model API for handwriting.
- Complex layouts — multi-column documents with mixed text and images may produce interleaved output. Use PSM 4 or PSM 3 for best results, or pre-crop the image to the text region.
- Large PDFs — very large PDFs (50+ pages) at high DPI (400+) can consume significant memory. Process in batches or reduce DPI for large files.
- Per-word confidence for PDFs — the word confidence table shows aggregate stats for PDFs. Per-word breakdown is only available for single image inputs.
- Windows only — path configuration uses Windows conventions. Linux/macOS users should update tesseract\_cmd to the output of  which tesseract  and remove the POPPLER\_PATH setting (Poppler is typically on PATH on those platforms).

|OCR Studio — Final Edition  ·  Built with Streamlit, Tesseract 5, OpenCV 4, Pillow, and pdf2image|
| :-: |

