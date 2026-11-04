from __future__ import annotations

import pymupdf

from medical_rag.ingest.bnf_layout import bnf_pages


def test_bnf_layout_tracks_page_hierarchy_fields_and_colored_dose_panel():
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 30), "1264 Pain and inflammation in musculoskeletal disorders", fontsize=9)
    page.insert_text((500, 30), "BNF 85", fontsize=9)
    page.insert_text((585, 300), "Musculoskeletal system", fontsize=9, rotate=90)
    page.insert_text((575, 140), "10", fontsize=10)
    panel = (224 / 255, 231 / 255, 245 / 255)
    page.draw_rect((50, 125, 285, 310), color=None, fill=panel)
    page.insert_text((60, 80), "10.1 Pain and inflammation", fontsize=16)
    page.insert_text((60, 110), "Naproxen", fontsize=14, color=(0, 0.45, 0.75))
    page.insert_text((230, 110), "22-Nov-2021", fontsize=7)
    page.insert_text((60, 145), "INDICATIONS AND DOSE", fontsize=9, color=(0, 0.45, 0.75))
    page.insert_text((60, 165), "Pain and inflammation in rheumatic disease", fontsize=9)
    page.insert_text((60, 188), "BY MOUTH", fontsize=9)
    page.insert_text((72, 205), "Adult: 0.5-1 g daily in 1-2 divided doses", fontsize=9)
    page.insert_text((60, 225), "Pain and inflammation with severe", fontsize=9)
    page.insert_text((60, 235), "symptoms", fontsize=9)
    page.insert_text((60, 245), "BY MOUTH", fontsize=9)
    page.insert_text((72, 265), "Adult: 1 g twice daily", fontsize=9)
    # Unlabelled dosage panel: same fill, but route and population-dose evidence.
    page.draw_rect((330, 38, 555, 128), color=None, fill=panel)
    page.insert_text((340, 50), "Acute drug-induced dystonic reactions", fontsize=9)
    page.insert_text((340, 70), "BY INTRAVENOUS INJECTION", fontsize=9)
    page.insert_text((340, 90), "Adult: 5-10 mg after 10 minutes", fontsize=9)
    # Same-colour non-dose callout must not be classified as a dose panel.
    page.draw_rect((330, 235, 555, 325), color=None, fill=panel)
    page.insert_text((340, 250), "IMPORTANT SAFETY INFORMATION", fontsize=9)
    page.insert_text((340, 270), "Monitor patients during treatment.", fontsize=9)
    page.insert_text((330, 340), "Renal impairment guidance outside the dose panel.", fontsize=9)

    fake_panel_tables = lambda _page: [
        (pymupdf.Rect(50, 125, 285, 310), "| Fake | table |\n|---|---|\n| dose | row |"),
        (pymupdf.Rect(330, 38, 555, 128), "| Fake | table |\n|---|---|\n| dose | row |"),
    ]
    tags = bnf_pages(doc, [0], body_size=9, tables_fn=fake_panel_tables)
    doc.close()

    title = next(tag for tag in tags if tag.kind == "title")
    assert title.ctx.drug == "Naproxen"
    assert title.ctx.chapter_no == "10"
    assert title.ctx.chapter == "Musculoskeletal system"
    assert title.ctx.subsection == "10.1 Pain and inflammation"
    assert title.ctx.printed_page == "1264"

    dose = [tag for tag in tags if tag.ctx.field == "INDICATIONS AND DOSE"]
    assert any(tag.ctx.indication == "Pain and inflammation in rheumatic disease" for tag in dose)
    assert any(tag.ctx.route == "BY MOUTH" for tag in dose)
    assert any(
        tag.ctx.indication == "Pain and inflammation with severe symptoms"
        and "Adult: 1 g twice daily" in tag.text
        for tag in dose
    )
    assert any("Adult: 0.5-1 g" in tag.text and tag.dose_box for tag in dose)
    assert not any("Fake" in tag.text for tag in tags)
    assert any(
        tag.ctx.indication == "Acute drug-induced dystonic reactions"
        and tag.ctx.route == "BY INTRAVENOUS INJECTION"
        and "Adult: 5-10 mg" in tag.text
        and tag.dose_box
        for tag in dose
    )
    assert not any("IMPORTANT SAFETY INFORMATION" in tag.text and tag.dose_box for tag in tags)
    assert any("Renal impairment guidance" in tag.text and not tag.dose_box for tag in tags)

def test_bnf_chunk_header_and_key_keep_entity_and_field_context():
    from medical_rag.ingest.bnf_chunking import chunk_header, group_key

    meta = {
        "h1": "Musculoskeletal system",
        "h2": "10.1 Pain and inflammation",
        "entity": "Naproxen",
        "preparation": "NAPROXEN 250MG TABLETS",
        "field": "INDICATIONS AND DOSE",
        "indication": "Pain and inflammation in rheumatic disease",
        "route": "BY MOUTH",
        "dose_box": True,
    }
    assert group_key(meta) != group_key({**meta, "entity": "Diclofenac"})
    assert group_key(meta) != group_key({**meta, "preparation": "NAPROXEN 500MG TABLETS"})
    header = chunk_header(meta)
    assert "Naproxen" in header
    assert "NAPROXEN 250MG TABLETS" in header
    assert "INDICATIONS AND DOSE" in header
    assert "BY MOUTH" in header


def test_stitch_keeps_different_preparations_separate():
    from medical_rag.ingest.bnf_layout import Ctx, Tagged, stitch

    ctx = Ctx(
        drug="Mesalazine",
        field="INDICATIONS AND DOSE",
        preparation="ASACOL MR 400MG TABLETS",
    )
    first = Tagged("Adult: 400 mg once daily", "text", 60, ctx, True)
    second_ctx = Ctx(
        drug="Mesalazine",
        field="INDICATIONS AND DOSE",
        preparation="ASACOL MR 800MG TABLETS",
    )
    second = Tagged("adult dose continuation", "text", 60, second_ctx, True)

    assert len(stitch([first, second])) == 2


def test_interaction_items_keep_pair_metadata_and_distinct_chunk_keys():
    from medical_rag.ingest.bnf_layout import BnfTagger, LBlock
    from medical_rag.ingest.bnf_chunking import group_key

    tagger = BnfTagger(body_size=8)
    tagger.c.field = "INTERACTIONS"
    a = tagger._emit_body(
        "Imatinib is predicted to increase the exposure to abemaciclib.",
        "text", 1, LBlock(30, 40, 400, 50, ""),
    )[0]
    b = tagger._emit_body(
        "Cobicistat is predicted to increase the exposure to abemaciclib.",
        "text", 1, LBlock(30, 60, 400, 70, ""),
    )[0]
    assert a.ctx.interaction_source == "Imatinib"
    assert a.ctx.interaction_target == "abemaciclib"
    assert group_key({"field": "INTERACTIONS", "interaction_id": a.ctx.interaction_id}) != group_key(
        {"field": "INTERACTIONS", "interaction_id": b.ctx.interaction_id}
    )


def test_chunk_page_label_prefers_printed_page():
    from medical_rag.models import Chunk

    chunk = Chunk(
        chunk_id="id", doc_id="doc", text="dose",
        page_start=1282, page_end=1282, title="BNF",
        metadata={"printed_page": "1263"},
    )
    assert chunk.page_label() == "p.1263"
    assert chunk.citation_label() == "BNF, p.1263"


def test_bnf_columns_follow_requested_right_first_order():
    from medical_rag.ingest.bnf_layout import LBlock, order_blocks

    blocks = [
        LBlock(45, 50, 200, 60, "left top"),
        LBlock(45, 120, 200, 130, "left bottom"),
        LBlock(330, 50, 500, 60, "right top"),
        LBlock(330, 120, 500, 130, "right bottom"),
    ]
    assert [b.text for b in order_blocks(blocks, 595)] == [
        "right top", "right bottom", "left top", "left bottom"
    ]


def test_next_page_context_comes_from_end_of_right_column():
    from medical_rag.ingest.bnf_layout import BnfTagger, LBlock

    tagger = BnfTagger(body_size=9)
    blocks = [
        LBlock(330, 80, 480, 100, "Naproxen", size=16, color=0x0071BC),
        LBlock(330, 120, 490, 132, "Right column continuation", size=9),
        LBlock(45, 200, 180, 220, "Nabumetone", size=16, color=0x0071BC),
    ]
    tagger.feed(0, blocks, 595)
    assert tagger.c.drug == "Naproxen"


def test_running_title_change_clears_stale_numbered_subsection():
    from medical_rag.ingest.bnf_layout import BnfTagger, LBlock

    tagger = BnfTagger(body_size=9)
    tagger.c.chapter = "Musculoskeletal system"
    tagger.c.subchapter = "Myasthenia gravis and Lambert-Eaton myasthenic syndrome"
    tagger.c.subsection = "3.2 Myasthenia gravis and"
    tagger.c.drug = "Neostigmine"
    tagger.c.field = "INDICATIONS AND DOSE"

    blocks = [
        LBlock(60, 10, 410, 22, "1264 Pain and inflammation in musculoskeletal disorders", zone="header"),
        LBlock(500, 10, 555, 22, "BNF 85", zone="header"),
        LBlock(570, 140, 580, 150, "10", zone="side"),
        LBlock(580, 180, 592, 280, "Musculoskeletal system", zone="side"),
        LBlock(330, 60, 450, 80, "Naproxen", size=15, color=0x0071BC),
        LBlock(330, 100, 520, 112, "Adult: Initially 750 mg", dose_box=True, drug_hint="Naproxen"),
    ]
    tagged = tagger.feed(1282, blocks, 595)

    naproxen_dose = next(item for item in tagged if "750 mg" in item.text)
    assert naproxen_dose.ctx.subchapter == "Pain and inflammation in musculoskeletal disorders"
    assert naproxen_dose.ctx.subsection is None
    assert naproxen_dose.ctx.drug == "Naproxen"


def test_shaded_blue_title_and_pale_body_are_a_table_not_a_dose_panel():
    from medical_rag.ingest.bnf_layout import extract_blocks, merge_tables, shaded_bnf_tables

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    blue = (199 / 255, 213 / 255, 238 / 255)
    pale = (225 / 255, 232 / 255, 245 / 255)
    page.draw_rect((40, 30, 555, 70), color=None, fill=blue)
    page.draw_rect((40, 70, 555, 150), color=None, fill=pale)
    page.insert_text((50, 45), "TABLE 2", fontsize=8, color=(0, 0.44, 0.74))
    page.insert_text((50, 59), "Drugs that cause nephrotoxicity", fontsize=9, color=(0, 0.44, 0.74))
    page.insert_text((50, 68), "This is a list of drugs.", fontsize=7)
    page.insert_text((50, 90), "aceclofenac", fontsize=8)
    page.insert_text((200, 90), "cefoxitin", fontsize=8)
    page.insert_text((350, 90), "diclofenac", fontsize=8)
    blocks = extract_blocks(page)
    tables = shaded_bnf_tables(page, blocks)
    assert len(tables) == 1
    merged = merge_tables(blocks, tables)
    table = next(b for b in merged if b.kind == "table")
    assert table.table_id == "2"
    assert table.table_title == "Drugs that cause nephrotoxicity"
    assert "aceclofenac" in table.text and "cefoxitin" in table.text and "diclofenac" in table.text
    assert not any(b.dose_box for b in merged)
    tags = bnf_pages(doc, [0], body_size=9)
    parsed_table = next(tag for tag in tags if tag.kind == "table")
    assert parsed_table.ctx.field == "TABLE"
    assert parsed_table.ctx.table_title == "Drugs that cause nephrotoxicity"
    doc.close()
