"""Audit script for Oxford Handbook of Clinical Medicine (OHCM) PDF layout."""
from __future__ import annotations

import re
from collections import Counter
import pymupdf

PDF_PATH = r"RAW_DATA\[Oxford Medical Handbooks] Ian Wilkinson, Tim Raine, Kate Wiles, Anna Goodhart, Catriona Ha - Oxford Handbook of Clinical Medicine (2017, Oxford University Press) - libgen.li.pdf"

def audit_doc():
    doc = pymupdf.open(PDF_PATH)
    print(f"Total PDF pages: {len(doc)}")

    # 1. Print TOC from PDF bookmarks
    toc = doc.get_toc()
    print("\n--- PDF Bookmarks (TOC) ---")
    for level, title, page in toc[:35]:
        print(f"L{level} | Page {page}: {title}")

    # 2. Inspect Table of Contents printed pages (around PDF index 6..8)
    print("\n--- Printed TOC Text Sample (PDF Page 7) ---")
    page_toc = doc.load_page(6) # 0-indexed page 6 = PDF page 7
    print(page_toc.get_text("text")[:1500])

    # 3. Inspect sample chapter pages (e.g., Chapter 3 Cardiovascular medicine: page index 104..110)
    print("\n--- Font & Style Analysis on Pages 105-110 ---")
    font_stats = Counter()
    color_stats = Counter()

    for page_idx in range(104, 115):
        page = doc.load_page(page_idx)
        print(f"\n=== PAGE INDEX {page_idx + 1} (rect: {page.rect}) ===")
        text_dict = page.get_text("dict")
        for b in text_dict["blocks"]:
            if b["type"] != 0: # text block
                continue
            for line in b["lines"]:
                spans = line["spans"]
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                first_span = spans[0]
                size = round(first_span["size"], 1)
                color = first_span["color"]
                font = first_span["font"]
                flags = first_span["flags"]
                font_stats[(font, size, flags, hex(color))] += 1
                color_stats[hex(color)] += 1

                # Print lines with large fonts, distinct colors, or potential headings/headers
                if size >= 10.0 or hex(color) not in ("0x0", "0x221f1f", "0x000000") or line.get("bbox")[1] < 50 or line.get("bbox")[1] > 700:
                    bbox = [round(v, 1) for v in line["bbox"]]
                    print(f"  bbox={bbox} | size={size} | font={font} | color={hex(color)} | text='{line_text}'")

    print("\n--- Top Font/Style Combinations across sample ---")
    for style, count in font_stats.most_common(20):
        print(f"Count {count:4d} | Font: {style[0]} | Size: {style[1]} | Flags: {style[2]} | Color: {style[3]}")

if __name__ == "__main__":
    audit_doc()
