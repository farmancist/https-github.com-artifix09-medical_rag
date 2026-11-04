"""Analyze structure, headers, titles, section headings, and index format of OHCM."""
import pymupdf
import re

PDF_PATH = r"RAW_DATA\[Oxford Medical Handbooks] Ian Wilkinson, Tim Raine, Kate Wiles, Anna Goodhart, Catriona Ha - Oxford Handbook of Clinical Medicine (2017, Oxford University Press) - libgen.li.pdf"

doc = pymupdf.open(PDF_PATH)

def analyze_page_structure(page_num):
    page = doc.load_page(page_num - 1)
    rect = page.rect
    w, h = rect.width, rect.height
    print(f"\n=======================================================")
    print(f"=== PAGE {page_num} (width={w:.1f}, height={h:.1f}) ===")
    print(f"=======================================================")

    text_dict = page.get_text("dict")
    lines_summary = []

    for b in text_dict["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            first_span = line["spans"][0]
            font = first_span["font"]
            size = round(first_span["size"], 1)
            color = hex(first_span["color"])
            bold = bool(first_span["flags"] & 2 or "Bold" in font or "bold" in font)
            bbox = [round(v, 1) for v in line["bbox"]]

            # Identify line position zone
            zone = "body"
            if bbox[1] < 35:
                zone = "header"
            elif bbox[1] > 480:
                zone = "footer"
            elif bbox[0] > 250 or bbox[2] < 35:
                zone = "side"

            lines_summary.append({
                "bbox": bbox,
                "text": text,
                "font": font,
                "size": size,
                "color": color,
                "bold": bold,
                "zone": zone
            })

    # Print zones
    headers = [l for l in lines_summary if l["zone"] == "header"]
    footers = [l for l in lines_summary if l["zone"] == "footer"]
    sides = [l for l in lines_summary if l["zone"] == "side"]
    body = [l for l in lines_summary if l["zone"] == "body"]

    print("\n[HEADERS]")
    for l in headers:
        print(f"  y={l['bbox'][1]} | size={l['size']} | c={l['color']} | {l['text']}")

    print("\n[FOOTERS]")
    for l in footers:
        print(f"  y={l['bbox'][1]} | size={l['size']} | c={l['color']} | {l['text']}")

    print("\n[BODY HEADINGS / TITLES (size >= 8.0 or bold or colored)]")
    for l in body:
        if l['size'] >= 7.8 or l['bold'] or l['color'] not in ('0x0', '0x221f1f', '0x231f20'):
            print(f"  y={l['bbox'][1]:5.1f} x={l['bbox'][0]:5.1f} | sz={l['size']:4.1f} | font={l['font']:25s} | c={l['color']:8s} | {l['text']}")

print("--- ANALYZING SAMPLE PAGES ---")
for p in [105, 106, 107, 108, 109, 110, 791, 881, 882]:
    analyze_page_structure(p)
