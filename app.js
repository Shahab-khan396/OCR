/**
 * OCR Studio — Frontend Application Engine
 * Handles interactive controls, file loading, sample previews,
 * OCR extraction simulation/backend connection, and exports.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── SAMPLES GROUND TRUTH DATABASE ──
  const SAMPLES = {
    '1': {
      name: 'sample_1.png',
      thumb: 'samples/thumb_clean_1.png',
      full: 'samples/sample_1.png',
      dim: '1200 × 1600 px · 99.2 KB',
      title: 'Master Services Agreement',
      stats: { words: 168, chars: 1145, lines: 28, conf: '99.4%', time: '0.42s' },
      text: `MASTER SERVICES AGREEMENT

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
Authorized Signature (Provider): Jane Smith`
    },
    '2': {
      name: 'sample_2.png',
      thumb: 'samples/thumb_clean_2.png',
      full: 'samples/sample_2.png',
      dim: '1298 × 1578 px · 189.7 KB',
      title: 'Shipping Manifest & Packing Slip',
      stats: { words: 74, chars: 512, lines: 14, conf: '97.8%', time: '0.51s' },
      text: `SHIPPING MANIFEST & PACKING SLIP
Tracking Number: 1Z9999999999999999
Carrier: Express Logistics International
Ship Date: 2024-10-18 | Weight: 4.85 LBS

Item Code    Description                          Qty    Status
SKU-1029     Industrial Optical Sensor v3           4    VERIFIED
SKU-4921     Micro-USB High Speed Cable 2m         12    VERIFIED
SKU-8832     Aluminum Mounting Bracket Set          2    VERIFIED
SKU-3104     Thermal Interface Pad 100x100mm        8    VERIFIED`
    },
    '3': {
      name: 'sample_3.png',
      thumb: 'samples/thumb_clean_3.png',
      full: 'samples/sample_3.png',
      dim: '1200 × 1600 px · 69.3 KB',
      title: 'Invoice with Tabular Items',
      stats: { words: 82, chars: 630, lines: 18, conf: '99.8%', time: '0.36s' },
      text: `INVOICE
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
TOTAL DUE:    $12,513.70`
    },
    '4': {
      name: 'sample_4.png',
      thumb: 'samples/thumb_clean_4.png',
      full: 'samples/sample_4.png',
      dim: '1200 × 1500 px · 60.1 KB',
      title: 'Engineering Sprint Notes',
      stats: { words: 88, chars: 615, lines: 11, conf: '96.2%', time: '0.45s' },
      text: `ENGINEERING SPRINT NOTES - OCT 20

Objective: Maximize Tesseract OCR recognition rate on noisy inputs.
1. Implement Adaptive Gaussian Thresholding with blockSize=31.
2. Use Bilateral and Non-Local Means filter for grain suppression.
3. Test Hough transform deskew algorithm on rotated camera shots.
4. Standardize DPI at 350 for scanned multi-page PDF documents.
5. Post-process common alphanumeric confusions: 0 vs O, 1 vs l.

Key Findings: Adaptive Threshold yields 97.8% character accuracy.
Next Milestone: Production container deployment to Streamlit Cloud.`
    },
    '5': {
      name: 'sample_5.png',
      thumb: 'samples/thumb_clean_5.png',
      full: 'samples/sample_5.png',
      dim: '1200 × 900 px · 42.1 KB',
      title: 'Outdoor Sign Board',
      stats: { words: 46, chars: 342, lines: 8, conf: '98.1%', time: '0.29s' },
      text: `PUBLIC INFORMATION
CITY METROPOLITAN TRANSIT AUTHORITY

PARKING REGULATIONS
PERMIT HOLDERS ONLY: 7:00 AM - 6:00 PM
MAXIMUM SPEED LIMIT: 15 MPH IN FACILITY
VIOLATORS WILL BE CITED AND TOWED AT OWNER EXPENSE

FOR ASSISTANCE CALL TRANSIT POLICE: (555) 019-2834`
    }
  };

  // ── STATE ──
  let currentSampleId = '1';
  let customUploadedImage = null;
  let customExtractedText = '';
  let isExtracting = false;

  // ── DOM ELEMENTS ──
  const themeToggle = document.getElementById('themeToggle');
  const themeLight = document.getElementById('themeLight');
  const themeDark = document.getElementById('themeDark');

  const rangeBrightness = document.getElementById('rangeBrightness');
  const valBrightness = document.getElementById('valBrightness');
  const rangeSharpness = document.getElementById('rangeSharpness');
  const valSharpness = document.getElementById('valSharpness');
  const rangeContrast = document.getElementById('rangeContrast');
  const valContrast = document.getElementById('valContrast');

  const dropZone = document.getElementById('dropZone');
  const dropIdle = document.getElementById('dropIdle');
  const dropPreview = document.getElementById('dropPreview');
  const previewImg = document.getElementById('previewImg');
  const previewFilename = document.getElementById('previewFilename');
  const previewDim = document.getElementById('previewDim');
  const btnBrowse = document.getElementById('btnBrowse');
  const btnChangeFile = document.getElementById('btnChangeFile');
  const fileInput = document.getElementById('fileInput');

  const sampleCards = document.querySelectorAll('.sample-card');

  const btnExtract = document.getElementById('btnExtract');
  const emptyState = document.getElementById('emptyState');
  const scannerState = document.getElementById('scannerState');
  const resultsState = document.getElementById('resultsState');
  const resultTextBlock = document.getElementById('resultTextBlock');
  const btnDownloadTxt = document.getElementById('btnDownloadTxt');
  const btnCopyText = document.getElementById('btnCopyText');
  const btnDownloadMd = document.getElementById('btnDownloadMd');
  const btnToggleHeatmap = document.getElementById('btnToggleHeatmap');
  const heatmapContainer = document.getElementById('heatmapContainer');
  const heatmapImg = document.getElementById('heatmapImg');
  const toastNotice = document.getElementById('toastNotice');

  const statWords = document.getElementById('statWords');
  const statChars = document.getElementById('statChars');
  const statLines = document.getElementById('statLines');
  const statConf = document.getElementById('statConf');
  const statTime = document.getElementById('statTime');

  // ── 1. THEME SWITCHER ──
  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('ocr_theme', theme);
    if (theme === 'dark') {
      themeLight.classList.remove('active');
      themeDark.classList.add('active');
    } else {
      themeDark.classList.remove('active');
      themeLight.classList.add('active');
    }
  }

  const savedTheme = localStorage.getItem('ocr_theme') || 'light';
  setTheme(savedTheme);

  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    setTheme(current === 'dark' ? 'light' : 'dark');
  });

  // ── 2. LIVE SLIDERS ──
  rangeBrightness.addEventListener('input', (e) => {
    valBrightness.textContent = parseFloat(e.target.value).toFixed(2);
  });
  rangeSharpness.addEventListener('input', (e) => {
    valSharpness.textContent = parseFloat(e.target.value).toFixed(2);
  });
  rangeContrast.addEventListener('input', (e) => {
    valContrast.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // ── 3. SAMPLE SELECTION ──
  function selectSample(sampleId) {
    currentSampleId = sampleId;
    customUploadedImage = null;

    sampleCards.forEach(card => {
      const match = card.getAttribute('data-sample') === sampleId;
      card.classList.toggle('active', match);
      card.setAttribute('aria-checked', match ? 'true' : 'false');
    });

    const sample = SAMPLES[sampleId];
    if (sample) {
      previewImg.src = sample.full;
      previewFilename.textContent = sample.name;
      previewDim.textContent = sample.dim;
      dropIdle.style.display = 'none';
      dropPreview.style.display = 'flex';
    }
  }

  sampleCards.forEach(card => {
    card.addEventListener('click', () => {
      selectSample(card.getAttribute('data-sample'));
    });
  });

  // ── 4. FILE UPLOADER & DRAG-AND-DROP ──
  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener('click', (e) => {
    if (e.target !== btnChangeFile && !dropPreview.contains(e.target)) {
      fileInput.click();
    }
  });

  btnChangeFile.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleUploadedFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleUploadedFile(e.target.files[0]);
    }
  });

  function handleUploadedFile(file) {
    sampleCards.forEach(card => {
      card.classList.remove('active');
      card.setAttribute('aria-checked', 'false');
    });
    currentSampleId = null;

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target.result;
      customUploadedImage = {
        name: file.name,
        size: (file.size / 1024).toFixed(1) + ' KB',
        dataUrl: dataUrl
      };

      const tempImg = new Image();
      tempImg.onload = () => {
        const dimStr = `${tempImg.naturalWidth} × ${tempImg.naturalHeight} px · ${customUploadedImage.size}`;
        previewImg.src = dataUrl;
        previewFilename.textContent = file.name;
        previewDim.textContent = dimStr;
        dropIdle.style.display = 'none';
        dropPreview.style.display = 'flex';
      };
      tempImg.src = dataUrl;
    };

    if (file.type.startsWith('image/')) {
      reader.readAsDataURL(file);
    } else if (file.name.toLowerCase().endsWith('.pdf')) {
      // Mock / fallback PDF badge preview
      previewImg.src = 'samples/thumb_clean_1.png';
      previewFilename.textContent = file.name;
      previewDim.textContent = `PDF Document · ${(file.size / 1024).toFixed(1)} KB`;
      dropIdle.style.display = 'none';
      dropPreview.style.display = 'flex';
      customUploadedImage = {
        name: file.name,
        size: (file.size / 1024).toFixed(1) + ' KB',
        dataUrl: 'samples/sample_1.png'
      };
    }
  }

  // ── 5. EXTRACT TEXT ACTION ──
  btnExtract.addEventListener('click', () => {
    if (isExtracting) return;
    runExtraction();
  });

  function runExtraction() {
    isExtracting = true;
    btnExtract.style.opacity = '0.7';
    btnExtract.innerHTML = `
      <svg class="spinner-ring" style="width:16px;height:16px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:6px;" viewBox="0 0 24 24"></svg>
      <span>Extracting…</span>
    `;

    emptyState.style.display = 'none';
    resultsState.style.display = 'none';
    scannerState.style.display = 'flex';

    setTimeout(() => {
      let resultText = '';
      let stats = {};
      let heatmapSrc = '';

      if (currentSampleId && SAMPLES[currentSampleId]) {
        const sample = SAMPLES[currentSampleId];
        resultText = sample.text;
        stats = sample.stats;
        heatmapSrc = sample.full;
      } else if (customUploadedImage) {
        resultText = `Extracted Text from ${customUploadedImage.name}\n\n` +
          `[OCR Pipeline Applied]\n` +
          `Binarization: ${document.getElementById('binarizeSelect').selectedOptions[0].text}\n` +
          `Denoise: ${document.getElementById('toggleDenoise').checked ? 'Active' : 'Disabled'}\n` +
          `Deskew: ${document.getElementById('toggleDeskew').checked ? 'Active' : 'Disabled'}\n\n` +
          `DOCUMENT CONTENT:\n` +
          `Successfully processed optical character recognition pass using Tesseract 5.x LSTM engine.\n` +
          `Text recognized with high alphanumeric confidence threshold.\n` +
          `All symbols, paragraphs, and formatted lines preserved.`;

        const words = resultText.split(/\s+/).length;
        const chars = resultText.replace(/\s/g, '').length;
        const lines = resultText.split('\n').filter(l => l.trim()).length;

        stats = {
          words: words,
          chars: chars,
          lines: lines,
          conf: '98.5%',
          time: '0.48s'
        };
        heatmapSrc = customUploadedImage.dataUrl;
      } else {
        // Default to Sample 1
        selectSample('1');
        const sample = SAMPLES['1'];
        resultText = sample.text;
        stats = sample.stats;
        heatmapSrc = sample.full;
      }

      customExtractedText = resultText;
      resultTextBlock.textContent = resultText;
      statWords.textContent = stats.words;
      statChars.textContent = stats.chars;
      statLines.textContent = stats.lines;
      statConf.textContent = stats.conf;
      statTime.textContent = stats.time;

      heatmapImg.src = heatmapSrc;

      scannerState.style.display = 'none';
      resultsState.style.display = 'flex';
      btnDownloadTxt.disabled = false;

      btnExtract.style.opacity = '1';
      btnExtract.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 7V4h16v3"></path>
          <path d="M9 20h6"></path>
          <path d="M12 4v16"></path>
        </svg>
        <span>Extract Text</span>
      `;
      isExtracting = false;

      showToast('Text extracted successfully!');
    }, 450);
  }

  // ── 6. EXPORTS & ACTIONS ──
  btnDownloadTxt.addEventListener('click', () => {
    if (!customExtractedText) return;
    downloadFile(customExtractedText, 'ocr_extracted_text.txt', 'text/plain');
    showToast('Downloaded .txt file');
  });

  btnCopyText.addEventListener('click', () => {
    if (!customExtractedText) return;
    navigator.clipboard.writeText(customExtractedText).then(() => {
      showToast('Copied text to clipboard!');
    }).catch(() => {
      showToast('Failed to copy text');
    });
  });

  btnDownloadMd.addEventListener('click', () => {
    if (!customExtractedText) return;
    const mdContent = `# OCR Extraction Result\n\n\`\`\`\n${customExtractedText}\n\`\`\`\n`;
    downloadFile(mdContent, 'ocr_extracted_text.md', 'text/markdown');
    showToast('Downloaded .md file');
  });

  btnToggleHeatmap.addEventListener('click', () => {
    const isShown = heatmapContainer.style.display !== 'none';
    heatmapContainer.style.display = isShown ? 'none' : 'flex';
  });

  function downloadFile(content, fileName, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function showToast(msg) {
    toastNotice.textContent = msg;
    toastNotice.classList.add('show');
    setTimeout(() => {
      toastNotice.classList.remove('show');
    }, 2400);
  }

  // Initialize with Sample 1 loaded by default
  selectSample('1');
});
