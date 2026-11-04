"""Analyze OHCM heading hierarchy across many pages."""
import pymupdf
import re
from collections import Counter

PDF_PATH = r"RAW_DATA\[Oxford Medical Handbooks] Ian Wilkinson, Tim Raine, Kate Wiles, Anna Goodhart, Catriona Ha - Oxford Handbook of Clinical Medicine (2017, Oxford University Press) - libgen.li.pdf"

doc = pymupdf.open(PDF_PATH)

def analyze_pages(page_nums):
    for p_num in page_nums:
        page = doc.load_page(p_num - 1)
        print(f"\n{'='*70}")
        print(f"=== PAGE {p_num} ===")
        print(f"{'='*70}")

        text_dict = page.get_text("dict")
        for b in text_dict["blocks"]:
            if b["type"] != 0: continue
            for line in b["lines"]:
                text = "".join(s["text"] for s in line["spans"]).strip()
                if not text: continue
                s = line["spans"][0]
                font = s["font"]
                size = round(s["size"], 1)
                color = hex(s["color"])
                bold = bool(s["flags"] & 2 or "Bold" in font)
                italic = bool(s["flags"] & 1 or "Italic" in font or "Ital" in font)
                bbox = [round(v, 1) for v in line["bbox"]]

                zone = "body"
                if bbox[1] < 35: zone = "HEADER"
                elif bbox[1] > 480: zone = "footer"

                print(f"  [{zone:6s}] y={bbox[1]:5.1f} x={bbox[0]:5.1f} | sz={size:4.1f} | font={font:25s} | c={color:8s} | bold={bold!s:5s} it={italic!s:5s} | {text}")

# Analyze pages across different chapters
print("### CHAPTER 3 - Cardiovascular medicine")
analyze_pages([105, 106, 107, 108, 109, 110, 111, 112, 113, 114])

print("\n\n### CHAPTER 4 - Chest medicine")
analyze_pages([173, 174, 175, 176])

print("\n\n### CHAPTER 19 - Emergencies")
analyze_pages([791, 792, 793])

print("\n\n### INDEX")
analyze_pages([881, 882, 883])

print("\n\n### EARLY WARNING SCORE / CARDIAC ARREST")
analyze_pages([905, 906, 907, 908])