import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

os.makedirs('samples', exist_ok=True)

# 1. Sample 1: Service Agreement Document
def make_sample_1():
    img = Image.new('RGB', (1200, 1600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype('arialbd.ttf', 36)
        font_sub = ImageFont.truetype('arialbd.ttf', 24)
        font_body = ImageFont.truetype('arial.ttf', 20)
        font_small = ImageFont.truetype('arial.ttf', 16)
    except Exception:
        font_title = font_sub = font_body = font_small = ImageFont.load_default()
    
    draw.text((600, 100), 'MASTER SERVICES AGREEMENT', fill=(20, 20, 20), font=font_title, anchor='mm')
    draw.line([(150, 135), (1050, 135)], fill=(180, 180, 180), width=2)
    
    text = (
        "This Master Services Agreement (\"Agreement\") is entered into as of October 20, 2024\n"
        "by and between Apex Technologies Inc. (\"Client\") and Quantum Cloud Solutions LLC (\"Provider\").\n\n"
        "1. SCOPE OF SERVICES\n"
        "Provider shall perform optical character recognition (OCR) and high-throughput document\n"
        "processing pipelines as detailed in applicable Statements of Work. Provider guarantees an\n"
        "accuracy baseline of not less than 99.2% on standard alphanumeric document scans.\n\n"
        "2. TERM AND TERMINATION\n"
        "The term of this Agreement shall commence on the Effective Date and shall continue\n"
        "for a period of twenty-four (24) months unless terminated earlier in accordance with the terms.\n\n"
        "3. CONFIDENTIALITY & DATA SECURITY\n"
        "All client files, images, PDFs, and extracted metadata shall remain the sole proprietary\n"
        "property of the Client. Provider shall adhere to SOC-2 Type II standards and AES-256 encryption\n"
        "for data in transit and at rest.\n\n"
        "4. WARRANTIES & LIABILITY\n"
        "Provider warrants that all professional services delivered shall conform to industry\n"
        "best practices and rigorous automated quality assurance metrics."
    )
    
    y = 180
    for block in text.split('\n\n'):
        if block.startswith(('1.', '2.', '3.', '4.')):
            lines = block.split('\n')
            draw.text((150, y), lines[0], fill=(20, 20, 20), font=font_sub)
            y += 35
            for l in lines[1:]:
                draw.text((150, y), l, fill=(40, 40, 40), font=font_body)
                y += 28
            y += 25
        else:
            for l in block.split('\n'):
                draw.text((150, y), l, fill=(40, 40, 40), font=font_body)
                y += 28
            y += 25

    # Signatures
    y = 1350
    draw.line([(150, y), (500, y)], fill=(80, 80, 80), width=1)
    draw.line([(700, y), (1050, y)], fill=(80, 80, 80), width=1)
    draw.text((150, y + 10), 'Authorized Signature (Client): John Doe', fill=(40, 40, 40), font=font_small)
    draw.text((700, y + 10), 'Authorized Signature (Provider): Jane Smith', fill=(40, 40, 40), font=font_small)
    
    img.save('samples/sample_1.png')
    thumb = img.resize((120, 150), Image.Resampling.LANCZOS)
    thumb.save('samples/thumb_clean_1.png')

# 2. Sample 2: Skewed Document (with deskew challenge)
def make_sample_2():
    img = Image.new('RGB', (1200, 1500), color=(250, 249, 246))
    draw = ImageDraw.Draw(img)
    try:
        font_h = ImageFont.truetype('arialbd.ttf', 32)
        font_b = ImageFont.truetype('arial.ttf', 20)
    except Exception:
        font_h = font_b = ImageFont.load_default()
    
    draw.text((100, 80), 'SHIPPING MANIFEST & PACKING SLIP', fill=(30, 30, 30), font=font_h)
    draw.text((100, 130), 'Tracking Number: 1Z9999999999999999', fill=(70, 70, 70), font=font_b)
    draw.text((100, 165), 'Carrier: Express Logistics International', fill=(70, 70, 70), font=font_b)
    draw.text((100, 200), 'Ship Date: 2024-10-18 | Weight: 4.85 LBS', fill=(70, 70, 70), font=font_b)
    
    draw.line([(100, 240), (1100, 240)], fill=(150, 150, 150), width=2)
    
    items = [
        ('Item Code', 'Description', 'Qty', 'Status'),
        ('SKU-1029', 'Industrial Optical Sensor v3', '4', 'VERIFIED'),
        ('SKU-4921', 'Micro-USB High Speed Cable 2m', '12', 'VERIFIED'),
        ('SKU-8832', 'Aluminum Mounting Bracket Set', '2', 'VERIFIED'),
        ('SKU-3104', 'Thermal Interface Pad 100x100mm', '8', 'VERIFIED'),
    ]
    y = 260
    for code, desc, q, stat in items:
        f = font_h if y == 260 else font_b
        draw.text((100, y), code, fill=(30, 30, 30), font=f)
        draw.text((320, y), desc, fill=(30, 30, 30), font=f)
        draw.text((800, y), q, fill=(30, 30, 30), font=f)
        draw.text((950, y), stat, fill=(30, 30, 30), font=f)
        y += 45
        draw.line([(100, y - 5), (1100, y - 5)], fill=(220, 220, 220), width=1)
    
    # Skew the image slightly by rotating 3.8 degrees
    rotated = img.rotate(3.8, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(240, 240, 238))
    rotated.save('samples/sample_2.png')
    thumb = rotated.resize((120, 150), Image.Resampling.LANCZOS)
    thumb.save('samples/thumb_clean_2.png')

# 3. Sample 3: Clean Invoice with table
def make_sample_3():
    img = Image.new('RGB', (1200, 1600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_logo = ImageFont.truetype('arialbd.ttf', 42)
        font_h = ImageFont.truetype('arialbd.ttf', 24)
        font_b = ImageFont.truetype('arial.ttf', 19)
        font_bold = ImageFont.truetype('arialbd.ttf', 19)
    except Exception:
        font_logo = font_h = font_b = font_bold = ImageFont.load_default()
    
    draw.text((100, 80), 'INVOICE', fill=(15, 118, 110), font=font_logo)
    draw.text((100, 140), "Invoice #: INV-2024-8841\nDate: October 20, 2024\nDue Date: Net 30 Days", fill=(70, 70, 70), font=font_b)
    
    draw.text((700, 80), 'BILLED TO:', fill=(100, 100, 100), font=font_h)
    draw.text((700, 120), "Acme Global Ventures\n742 Evergreen Terrace\nSpringfield, OR 97477", fill=(40, 40, 40), font=font_b)
    
    draw.rectangle([(100, 250), (1100, 300)], fill=(240, 245, 245))
    draw.text((120, 265), 'Description', fill=(30, 30, 30), font=font_bold)
    draw.text((650, 265), 'Hours / Qty', fill=(30, 30, 30), font=font_bold)
    draw.text((850, 265), 'Rate', fill=(30, 30, 30), font=font_bold)
    draw.text((1000, 265), 'Amount', fill=(30, 30, 30), font=font_bold)
    
    rows = [
        ('Cloud OCR Pipeline Integration', '40', '$125.00', '$5,000.00'),
        ('Computer Vision Preprocessing Module', '24', '$140.00', '$3,360.00'),
        ('High-Availability API Configuration', '16', '$150.00', '$2,400.00'),
        ('Technical Documentation & Training', '8', '$100.00', '$800.00'),
    ]
    y = 320
    for desc, qty, rate, amt in rows:
        draw.text((120, y), desc, fill=(40, 40, 40), font=font_b)
        draw.text((680, y), qty, fill=(40, 40, 40), font=font_b)
        draw.text((850, y), rate, fill=(40, 40, 40), font=font_b)
        draw.text((990, y), amt, fill=(40, 40, 40), font=font_b)
        y += 45
        draw.line([(100, y - 5), (1100, y - 5)], fill=(230, 230, 230), width=1)
    
    y += 40
    draw.text((750, y), 'Subtotal:', fill=(70, 70, 70), font=font_b)
    draw.text((980, y), '$11,560.00', fill=(40, 40, 40), font=font_b)
    y += 35
    draw.text((750, y), 'Tax (8.25%):', fill=(70, 70, 70), font=font_b)
    draw.text((980, y), '$953.70', fill=(40, 40, 40), font=font_b)
    y += 45
    draw.rectangle([(730, y - 5), (1100, y + 45)], fill=(240, 253, 250))
    draw.text((750, y + 5), 'TOTAL DUE:', fill=(15, 118, 110), font=font_h)
    draw.text((950, y + 5), '$12,513.70', fill=(15, 118, 110), font=font_h)
    
    img.save('samples/sample_3.png')
    thumb = img.resize((120, 150), Image.Resampling.LANCZOS)
    thumb.save('samples/thumb_clean_3.png')

# 4. Sample 4: Meeting Notes / Research memo
def make_sample_4():
    img = Image.new('RGB', (1200, 1500), color=(253, 252, 248))
    draw = ImageDraw.Draw(img)
    try:
        font_h = ImageFont.truetype('arialbd.ttf', 28)
        font_b = ImageFont.truetype('calibri.ttf', 22)
    except Exception:
        font_h = font_b = ImageFont.load_default()
    
    for ly in range(120, 1450, 40):
        draw.line([(80, ly), (1120, ly)], fill=(230, 235, 245), width=1)
    draw.line([(160, 60), (160, 1450)], fill=(245, 200, 200), width=2)
    
    draw.text((190, 80), 'ENGINEERING SPRINT NOTES - OCT 20', fill=(30, 40, 60), font=font_h)
    
    notes = [
        'Objective: Maximize Tesseract OCR recognition rate on noisy inputs.',
        '1. Implement Adaptive Gaussian Thresholding with blockSize=31.',
        '2. Use Bilateral and Non-Local Means filter for grain suppression.',
        '3. Test Hough transform deskew algorithm on rotated camera shots.',
        '4. Standardize DPI at 350 for scanned multi-page PDF documents.',
        '5. Post-process common alphanumeric confusions: 0 vs O, 1 vs l.',
        'Key Findings: Adaptive Threshold yields 97.8% character accuracy.',
        'Next Milestone: Production container deployment to Streamlit Cloud.'
    ]
    y = 170
    for note in notes:
        draw.text((190, y), note, fill=(35, 45, 65), font=font_b)
        y += 80
        
    img.save('samples/sample_4.png')
    thumb = img.resize((120, 150), Image.Resampling.LANCZOS)
    thumb.save('samples/thumb_clean_4.png')

# 5. Sample 5: Outdoor Sign Board
def make_sample_5():
    img = Image.new('RGB', (1200, 900), color=(220, 228, 225))
    draw = ImageDraw.Draw(img)
    try:
        font_sign_big = ImageFont.truetype('arialbd.ttf', 46)
        font_sign_mid = ImageFont.truetype('arialbd.ttf', 30)
        font_sign_small = ImageFont.truetype('arialbd.ttf', 22)
    except Exception:
        font_sign_big = font_sign_mid = font_sign_small = ImageFont.load_default()
    
    draw.rectangle([(150, 100), (1050, 800)], fill=(255, 255, 255), outline=(13, 148, 136), width=12)
    draw.rectangle([(170, 120), (1030, 780)], fill=(250, 250, 250), outline=(200, 210, 215), width=4)
    
    draw.rectangle([(174, 124), (1026, 260)], fill=(13, 148, 136))
    draw.text((600, 165), 'PUBLIC INFORMATION', fill=(255, 255, 255), font=font_sign_big, anchor='mm')
    draw.text((600, 220), 'CITY METROPOLITAN TRANSIT AUTHORITY', fill=(204, 251, 241), font=font_sign_small, anchor='mm')
    
    draw.text((600, 340), 'PARKING REGULATIONS', fill=(15, 23, 42), font=font_sign_mid, anchor='mm')
    draw.text((600, 420), 'PERMIT HOLDERS ONLY: 7:00 AM - 6:00 PM', fill=(51, 65, 85), font=font_sign_small, anchor='mm')
    draw.text((600, 480), 'MAXIMUM SPEED LIMIT: 15 MPH IN FACILITY', fill=(51, 65, 85), font=font_sign_small, anchor='mm')
    draw.text((600, 540), 'VIOLATORS WILL BE CITED AND TOWED AT OWNER EXPENSE', fill=(185, 28, 28), font=font_sign_small, anchor='mm')
    
    draw.line([(250, 620), (950, 620)], fill=(200, 200, 200), width=2)
    draw.text((600, 680), 'FOR ASSISTANCE CALL TRANSIT POLICE: (555) 019-2834', fill=(71, 85, 105), font=font_sign_small, anchor='mm')
    
    img.save('samples/sample_5.png')
    thumb = img.resize((120, 90), Image.Resampling.LANCZOS)
    thumb.save('samples/thumb_clean_5.png')

make_sample_1()
make_sample_2()
make_sample_3()
make_sample_4()
make_sample_5()
print('All samples generated successfully.')
