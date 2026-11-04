"""List every block matching the chapter regex, with geometry."""
import re
import pymupdf
from medical_rag.ingest.ohcm_layout import extract_ohcm_blocks

PDF = r"RAW_DATA\[Oxford Medical Handbooks] Ian Wilkinson, Tim Raine, Kate Wiles, Anna Goodhart, Catriona Ha - Oxford Handbook of Clinical Medicine (2017, Oxford University Press) - libgen.li.pdf"
CHAP = re.compile(r"^(\d+)\s+([A-Z][A-Za-z0-9 ,\-/()']{2,80})$")

doc = pymupdf.open(PDF)
for p in range(1, 911):
    page = doc.load_page(p - 1)
    blocks = extract_ohcm_blocks(page)
    for b in blocks:
        if CHAP.match(b["text"]):
            print(
                f"p{p}: y0={b['y0']:5.1f} x0={b['x0']:5.1f} sz={b['size']:4.1f} "
                f"bold={b['bold']!s:5s} it={b['italic']!s:5s} | {b['text'][:55]!r}"
            )