"""Inspect OHCM parser output: structure, sections, page numbers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_rag.ingest.pdf_parser import PDFParser

PDF = r"RAW_DATA\[Oxford Medical Handbooks] Ian Wilkinson, Tim Raine, Kate Wiles, Anna Goodhart, Catriona Ha - Oxford Handbook of Clinical Medicine (2017, Oxford University Press) - libgen.li.pdf"

parser = PDFParser()
doc = parser.parse(PDF)

print(f"Title: {doc.title}")
print(f"Pages: {doc.n_pages}")
print(f"Blocks: {len(doc.blocks)}")

# Show unique sections (chapter > topic > section)
sections = sorted({b.section for b in doc.blocks if b.section})
print(f"\nUnique sections ({len(sections)}):")
for s in sections[:40]:
    print(f"  {s}")

# Show sample blocks from first 3 chapters
print("\n--- Sample blocks from Chapter 3 (Cardiovascular, pages 105-110) ---")
count = 0
for b in doc.blocks:
    if b.page in range(105, 111):
        print(f"  p{b.page} | section={b.section!r} | {b.text[:100]}")
        count += 1
        if count > 25:
            break

# Check page numbers are preserved
print("\n--- Page number distribution (first 20 pages) ---")
for b in doc.blocks[:30]:
    print(f"  p{b.page} | {b.text[:80]}")