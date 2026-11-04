"""OHCM-aware layout analysis on top of PyMuPDF.

The Oxford Handbook of Clinical Medicine (10th ed.) has a distinctive
two-band header that no generic parser can read:

  band A  y0 < 25          running page number (GriffithGothic-Regular 7.5,
                            left or right edge) + the chapter's *topic title*
                            (GriffithGothic-Black 7.0-9.0, indented). On the
                            chapter's opening page the topic title sits in the
                            body instead, at y0 ~17-18.

  band B  25 <= y0 < 470   body text (GriffithGothic-Light 7.0) whose first
                            span may be a bold-italic label ("General inspection",
                            "• Rate:", "Reading an ECG") that is a sub-heading,
                            not a bullet.

Everything below y0 470 is the page footer (figure captions, running notes).

The tagger walks pages in order so the running topic title carries across
page breaks, exactly like the BNF monograph context carries across pages.

Font flags in this PDF are inverted from the usual PyMuPDF convention:
    flags=20  -> GriffithGothic-Black   (real bold)
    flags=22  -> GriffithGothic-BoldItal (bold+italic)
    flags= 6  -> GriffithGothic-RegularIt / LightItal (italic body)
    flags= 4  -> plain body
So "bold" is tested as (flags & 16) and "italic" as (flags & 1), not the
opposite -- otherwise italic body text is mistaken for a sub-heading and
Black headings are missed entirely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

OHCM_PARSER_VERSION = "ohcm-layout-1.0.1"

# Running page number: small, not bold, at a page edge, digits only.
_RUNNING_PAGE_RE = re.compile(r"^\d{1,4}$")
# Chapter heading: "3 Cardiovascular medicine", "19 Emergencies".
_CHAPTER_RE = re.compile(r"^(\d+)\s+([A-Z][A-Za-z0-9 ,\-/()']{2,80})$")
# The References section (ch. 20, pp. 865-880) repeats a mini contents list
# ("Contents" + "Chapter 1 Thinking about medicine 852" ...) in the body band.
# Every chapter's first page also carries a "Contents" label at y0 ~48.6. These
# are page furniture, not content.
_CHAPTER_LIST_RE = re.compile(r"^Chapter\s+\d+[:\s]")

# Pages before the first chapter: title page, contents, preface, symbols,
# the "Index to emergency topics" list. They have no running header and must
# not seed chapter/topic context.
_FRONT_MATTER_PAGES = frozenset(range(1, 13))


@dataclass
class OHCMCtx:
    chapter_title: str | None = None
    topic_title: str | None = None
    sub_header: str | None = None

    def label(self) -> str:
        parts = [self.chapter_title, self.topic_title, self.sub_header]
        return " > ".join(p for p in parts if p)


def extract_ohcm_blocks(page) -> list[dict]:
    """One dict per text line, with geometry and font flags.

    Spans are merged into a single line because the topic title and the
    sub-heading labels are often split across spans with different fonts
    ("Reading an ECG" = "Reading an " [Black 7.0] + "ECG" [Black 5.4]).
    """
    blocks: list[dict] = []
    text_dict = page.get_text("dict")
    for b in text_dict["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            first = spans[0]
            bbox = line["bbox"]
            blocks.append(
                {
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "text": text,
                    "size": round(first["size"], 1),
                    # Inverted flag convention for this PDF (see module docstring).
                    "bold": bool(first["flags"] & 16),
                    "italic": bool(first["flags"] & 1),
                }
            )
    return blocks


def _is_subheading(b: dict, nxt: dict | None) -> bool:
    """A Black label that opens a line and is followed by its body text.

    Real sub-headings ("General inspection", "• Rate:", "Reading an ECG")
    sit at the left margin and are immediately followed by a Light body
    line. Figure captions ("The randomized trial", "Fig 3.1 ...") are also
    Black but hang at the end of a page or float in a column with nothing
    below them -- requiring a body line underneath keeps them out of the
    sub-header slot.
    """
    if not (b["bold"] and not b["italic"] and b["x0"] < 24 and len(b["text"]) <= 60):
        return False
    # Bullets ("•", "○", "▸") are body text, not sub-headings.
    if b["text"][:1] in "•○▸►○●":
        return False
    if nxt is None:
        return False
    return (
        0 <= nxt["y0"] - b["y1"] < 16
        and not nxt["bold"]
        and abs(nxt["x0"] - b["x0"]) < 8
    )


def parse_ohcm_pages(doc, page_indices) -> list[dict]:
    """Tag every body line with its chapter / topic / sub-header context."""
    ctx = OHCMCtx()
    results: list[dict] = []
    for idx in page_indices:
        page = doc.load_page(idx)
        page_no = idx + 1
        front_matter = page_no in _FRONT_MATTER_PAGES
        # Pages 1-12 (title page, contents, preface, symbols, "Index to
        # emergency topics") have no running header and no chapter context;
        # their content is not clinical reference material.
        if front_matter:
            continue
        blocks = extract_ohcm_blocks(page)

        # The band-A topic title is the topmost block on the page. A short
        # bold label lower in the band (e.g. "• Colour:" on p52, "• Apex
        # beat." on p64) is a sub-list item, not a topic title.
        top_y0 = min((b["y0"] for b in blocks), default=0.0)

        for i, b in enumerate(blocks):
            y0, text = b["y0"], b["text"]

            # --- band A: running header (page number + topic title) ---------
            # The header band is tight: the running page number sits at
            # y0~18.9 and the chapter/topic titles at y0~16-18. Sub-list
            # items ("• Colour:", "• Apex beat.") also reach into y0~24
            # and must stay in the body, so the band stops at 22.
            if y0 < 22:
                if _RUNNING_PAGE_RE.match(text):
                    continue  # running page number, not content
                m = _CHAPTER_RE.match(text)
                # The chapter heading sits at the left margin (x0 ~14, size
                # 11.2). The numbered sub-section labels ("6 Face",
                # "4 Power", "1 Inspection") also match this regex but are
                # indented to x0 47-92 with size 6.3 -- they are
                # sub-headings, not chapters.
                if m and not text.isupper() and b["x0"] < 30 and b["size"] >= 10:
                    ctx.chapter_title = m.group(2)
                    ctx.topic_title = None
                    ctx.sub_header = None
                    continue
                # Topic title on a continuation page. GriffithGothic-Black
                # distinguishes it from the Light body font; the size varies
                # (7.0-9.0) so we key off the font, not the size. It sits at
                # the left margin (x0 <= 35). The numbered sub-section labels
                # ("6 Face", "4 Power", "1 Inspection") are also Black but
                # indented to x0 69-170 and must not be mistaken for topic
                # titles -- they are sub-headings for the body below.
                if (
                    b["bold"]
                    and not b["italic"]
                    and y0 <= top_y0 + 2
                    and b["x0"] < 50
                ):
                    ctx.topic_title = text
                continue

            # --- band B: body ------------------------------------------------
            if y0 < 470:
                # The chapter running title is repeated in the body band on
                # every page of chapters 1-2 (e.g. "History and examination"
                # at y0 56-68). It is page furniture, not content.
                if y0 < 80 and text.casefold() == (ctx.chapter_title or "").casefold():
                    continue
                # The References section (pp. 865-880) repeats a mini contents
                # list in the body band ("Contents", "Chapter 1 ... 852"),
                # and every chapter's first page carries a "Contents" label at
                # y0 ~48.6. These are page furniture, not content.
                if text.casefold() == "contents" or _CHAPTER_LIST_RE.match(text):
                    continue
                sub = _is_subheading(b, blocks[i + 1] if i + 1 < len(blocks) else None)
                if sub:
                    ctx.sub_header = text
                elif b["bold"]:
                    # A figure caption or numbered list item ("Fig 3.1 ...",
                    # "1 Irritative bladder symptoms:") ends the current
                    # sub-section's body flow, so the sub-header must not
                    # bleed onto the next page behind it.
                    ctx.sub_header = None
                results.append(
                    {
                        "text": text,
                        "page": page_no,
                        "ctx": replace(ctx),
                    }
                )
    return results