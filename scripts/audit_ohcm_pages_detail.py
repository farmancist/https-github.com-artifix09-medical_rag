"""Detailed layout auditor for OHCM PDF pages."""
from __future__ import annotations

import fitz

PDF_PATH = r"RAW_DATA\[Oxford Medical Handbooks] Ian Wilkinson, Tim Raine, Kate Wiles, Anna Goodhart, Catriona Ha - Oxford Handbook of Clinical Medicine (2017, Oxford University Press) - libgen.li.pdf"

def inspect_pages(pages):
    doc = fitz.open(PDF_PATH)
    for p_num in pages:
        page = doc.load_page(p_num - 1)
        print(f"\n=======================================================")
        print(f"=== PDF PAGE {p_num} (0-based index {p_num - 1}) ===")
        print(f"=======================================================")
        text_dict = page.get_text("dict")
        for b_idx, b in enumerate(text_dict["blocks"]):
            if b["type"] != 0:
                print(f"Block #{b_idx}: [Image/Drawing] bbox={b['bbox']}")
                continue
            for l_idx, line in enumerate(b["lines"]):
                spans_str = []
                for s in line["spans"]:
                    font = s["font"]
                    size = round(s["size"], 1)
                    color = hex(s["color"])
                    flags = s["flags"]
                    bold = bool(flags & 2 or "Bold" in font or "bold" in font)
                    italic = bool(flags & 1 or "Italic" in font or "Oblique" in font)
                    spans_str.append(f"[{s['text']!r} font={font} s={size} c={color} bold={bold} it={italic}]")
                bbox = [round(v, 1) for v in line["bbox"]]
                print(f"  Line {l_idx:2d} | bbox={bbox} | {' '.join(spans_str)}")

if __name__ == "__main__":
    # Let's inspect:
    # Page 105: Chapter 3 start (Cardiovascular)
    # Page 106: First topic in Chapter 3
    # Page 107: Next page
    # Page 791: Chapter 19 Emergencies start
    # Page 881: Index start
    inspect_pages([105, 106, 107, 791, 881])
