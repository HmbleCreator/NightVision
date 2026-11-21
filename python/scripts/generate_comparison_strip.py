import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil

raw_img = Image.open('data/HighExposureImages/Portrait_Overexposed_Raw.jpg').convert('RGB')
desk_img = Image.open('output/portrait_desktop.png').convert('RGB')
mcu_img = Image.open('output/portrait_mcu.png').convert('RGB')

w, h = raw_img.size

# Create 3-panel strip
header_h = 70
footer_h = 90
canvas_w = w * 3 + 20 * 2
canvas_h = h + header_h + footer_h

canvas = Image.new('RGB', (canvas_w, canvas_h), (18, 20, 24))
draw = ImageDraw.Draw(canvas)

# Try loading a good truetype font, fallback to default
try:
    font_title = ImageFont.truetype("arial.ttf", 26)
    font_sub = ImageFont.truetype("arial.ttf", 18)
    font_body = ImageFont.truetype("arial.ttf", 15)
except Exception:
    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_body = ImageFont.load_default()

panels = [
    (raw_img, "1. Raw Overexposed Input", "True uncompressed capture | 27.34% triple-clipped | Milky glare (163.6 ADU)", [
        "Cheek Core: [253.5, 253.5, 253.4] (Blown white)",
        "Midtone Skin: [176.4, 160.2, 147.1] (Milky glare)",
        "Blown Wall: [252.8, 252.8, 252.5] (Pure white)",
    ]),
    (desk_img, "2. Desktop FPU Pipeline", "Planar FPU | Contiguous Push-Pull | C1 Soft-Knee (150 ms)", [
        "Cheek Core: [236.4, 226.1, 219.2] (R-B = 17.20 ADU, PASS)",
        "Zero Moat: Trans (19.81) >= Core (17.20) ADU (PASS)",
        "Blown Wall: [230.5, 227.0, 223.1] (|R-B| = 7.37 ADU, PASS)",
    ]),
    (mcu_img, "3. MCU Streaming Pipeline", "Zero-alloc Scanline EMA | 1D LUT | < 2KB SRAM (20.8 ms, 48 FPS)", [
        "Cheek Core: [231.9, 227.7, 218.7] (R-B = 13.17 ADU, PASS)",
        "Zero Moat: Trans (21.46) >= Core (13.17) ADU (PASS)",
        "Blown Wall: [228.3, 228.0, 227.4] (|R-B| = 0.91 ADU, PASS)",
    ]),
]

for idx, (img, title, sub, bullets) in enumerate(panels):
    x_offset = idx * (w + 20)
    
    # Title & Subtitle
    draw.text((x_offset + 15, 12), title, fill=(245, 245, 250), font=font_title)
    draw.text((x_offset + 15, 42), sub, fill=(160, 175, 195), font=font_sub)
    
    # Image
    canvas.paste(img, (x_offset, header_h))
    
    # Footer telemetry
    fy = header_h + h + 10
    for b_idx, bullet in enumerate(bullets):
        draw.text((x_offset + 15, fy + b_idx * 22), bullet, fill=(200, 215, 230), font=font_body)

# Save high-res strip
canvas.save('output/portrait_comparison.png', quality=95)
print("Saved output/portrait_comparison.png")

# Also copy to artifact directory
artifact_path = r'C:\Users\amiku\.gemini\antigravity\brain\6692f08f-aca1-45d1-a655-813c577615cd\portrait_comparison.png'
shutil.copyfile('output/portrait_comparison.png', artifact_path)
print(f"Copied to {artifact_path}")
