"""Generate OG image for GitHub Pages / LinkedIn sharing."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

W, H = 1200, 630
bg = (17, 17, 17)
white = (255, 255, 255)
grey = (160, 160, 160)

img = Image.new("RGB", (W, H), bg)
draw = ImageDraw.Draw(img)

# Try system fonts for clean rendering
try:
    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 52)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    font_author = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except OSError:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()
    font_author = ImageFont.load_default()

# Main headline - result first
headline = "Your safety classifier is drifting.\nHere's how to know."
draw.multiline_text((80, 140), headline, fill=white, font=font_large, spacing=16)

# Subline
subline = "86.6% detection across 800 pre-registered factorial cells."
draw.multiline_text((80, 340), subline, fill=grey, font=font_small, spacing=10)

# Author + date
draw.text((80, 520), "Jun Wen Leong · June 2026", fill=grey, font=font_author)

out = Path("docs/assets/og-image.png")
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out, "PNG")
print(f"Saved {out}")
