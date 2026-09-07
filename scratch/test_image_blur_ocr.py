from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import pytesseract
import io

# 1. Generate a synthetic high-quality CBC snippet image
img = Image.new('RGB', (650, 220), color='white')
draw = ImageDraw.Draw(img)

text_lines = [
    "COMPLETE BLOOD COUNT (CBC)",
    "Hemoglobin: 8.2 g/dL   (Ref: 12.0 - 15.5 g/dL)",
    "Total WBC: 14500 /cumm (Ref: 4000 - 10000 /cumm)",
    "Platelets: 65000 /cumm (Ref: 150000 - 450000)",
    "Serum Creatinine: 2.4 mg/dL (Ref: 0.6 - 1.2 mg/dL)"
]

y = 20
for line in text_lines:
    draw.text((30, y), line, fill='black')
    y += 35

# 2. Intentionally degrade the image with motion blur and low contrast
# (simulating a blurry smartphone photo taken in dim clinic lighting)
blurry_img = img.filter(ImageFilter.GaussianBlur(radius=1.3))
enhancer = ImageEnhance.Contrast(blurry_img)
low_contrast_blurry = enhancer.enhance(0.7)

# Raw OCR on blurry image without enhancement
raw_blurry_ocr = pytesseract.image_to_string(low_contrast_blurry).strip()
print("=== 1. RAW OCR ON BLURRY / LOW-CONTRAST IMAGE (WITHOUT PIPELINE) ===")
print(raw_blurry_ocr if raw_blurry_ocr else "[EMPTY - Tesseract failed completely]")

# 3. Apply our Tier-1 Adaptive Restoration Pipeline:
# - Autocontrast normalization
# - Lanczos super-sampling upscale
# - Unsharp mask edge reconstruction
# - Adaptive contrast boost
gray = low_contrast_blurry.convert('L')
auto = ImageOps.autocontrast(gray, cutoff=2)
w, h = auto.size
scale = max(1.5, min(2.5, 1500 / max(w, h)))
upscaled = auto.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=2.0, percent=160, threshold=3))
final_enhanced = ImageEnhance.Contrast(sharpened).enhance(1.5)

restored_ocr = pytesseract.image_to_string(final_enhanced).strip()
print("\n=== 2. RESTORED OCR WITH TIER-1 ADAPTIVE PRE-PROCESSING ===")
print(restored_ocr)
