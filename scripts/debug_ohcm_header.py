"""Debug OHCM header detection logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_rag.ingest.ohcm_layout import extract_ohcm_blocks

import pymupdf

PDF = r"RAW_DATA\[Oxford Medical Handbooks] Ian Wilkinson, Tim Raine, Kate Wiles, Anna Goodhart, Catriona Ha - Oxford Handbook of Clinical Medicine (2017, Oxford University Press) - libgen.li.pdf"
doc = pymupdf.open(PDF)

for p in [105, 106, 107, 108, 109, 110]:
    page = doc.load_page(p - 1)
    blocks = extract_ohcm_blocks(page)
    print(f"\n=== PAGE {p} ===")
    for b in blocks:
        if b["y0"] < 50:
            print(f"  y0={b['y0']:5.1f} | sz={b['size']:4.1f} | bold={b['bold']!s:5s} it={b['italic']!s:5s} | text={b['text']!r}")