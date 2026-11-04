"""BNF-aware layout analysis on top of PyMuPDF.

Pipeline per page:
    extract_blocks()  -> geometry + style per block, zone = header/footer/side/body
    order_blocks()    -> two-column reading order with full-width "spanner" bands
Pipeline across pages (state carries over, because a monograph spans pages):
    BnfTagger.feed()  -> chapter / section / subsection / drug / field per block
    stitch()          -> rejoin sentences broken at a column or page boundary
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

BNF_PARSER_VERSION = "bnf-layout-1.1.0"

# --------------------------------------------------------------------------
# Closed vocabulary: every field label the BNF prints (legend pp. xii-xiii).
# --------------------------------------------------------------------------
LABELS = [
    "INDICATIONS AND DOSE", "CONTRA-INDICATIONS", "CAUTIONS", "CAUTIONS, FURTHER INFORMATION",
    "INTERACTIONS", "SIDE-EFFECTS", "SIDE-EFFECTS, FURTHER INFORMATION",
    "ALLERGY AND CROSS-SENSITIVITY", "CONCEPTION AND CONTRACEPTION", "PREGNANCY",
    "BREAST FEEDING", "HEPATIC IMPAIRMENT", "RENAL IMPAIRMENT", "PRE-TREATMENT SCREENING",
    "MONITORING REQUIREMENTS", "EFFECTS ON LABORATORY TESTS", "TREATMENT CESSATION",
    "DIRECTIONS FOR ADMINISTRATION", "PRESCRIBING AND DISPENSING INFORMATION",
    "HANDLING AND STORAGE", "PATIENT AND CARER ADVICE", "PROFESSION SPECIFIC INFORMATION",
    "NATIONAL FUNDING/ACCESS DECISIONS", "LESS SUITABLE FOR PRESCRIBING",
    "EXCEPTION TO LEGAL CATEGORY", "MEDICINAL FORMS", "DRUG ACTION", "UNLICENSED USE",
    "DOSE ADJUSTMENTS DUE TO INTERACTIONS", "DOSE EQUIVALENCE AND CONVERSION", "POTENCY",
    "PHARMACOKINETICS", "IMPORTANT SAFETY INFORMATION", "DOSES AT EXTREMES OF BODY-WEIGHT",
]
_LABEL_RE = re.compile(
    r"^[\W_]{0,3}(?:l\s+)?(" + "|".join(re.escape(x) for x in sorted(LABELS, key=len, reverse=True)) + r")(?![A-Za-z])"
)
_LEAD_JUNK_RE = re.compile(r"^[\u25b6\u25b8\u25ba\u25cf\u2022\u25aa\ue000-\uf8ff\s]+")
_PUA_RE = re.compile(r"[\ue000-\uf8ff]")
_DATE_RE = re.compile(r"\b(\d{1,2}-[A-Z][a-z]{2}-\d{4})\b")
_NUM_HEADING_RE = re.compile(r"^(\d{1,2})(?:\.(\d{1,2}))?([a-z])?\s+(\S.*)$", re.S)
_PAGE_TOKEN_RE = re.compile(r"\d{1,4}")
_ROMAN_RE = re.compile(r"[ivxlcdm]{1,6}", re.I)
_BNF_TAG_RE = re.compile(r"\bBNF\s*\d+\b", re.I)
_POP_RE = re.compile(r"^(Adult|Child|Neonate|Elderly|Infant|Initially|For |When )", re.I)
_FREQ_RE = re.compile(r"(?m)^(Common or very common|Uncommon|Rare or very rare|Frequency not known)\b[ \t]*")
_ROUTE_TAIL_RE = re.compile(r"^(?:INJECTION|INFUSION|INHALATION|ORAL|RECTAL|MOUTH|SKIN|EYE|EAR|VAGINA|TUBE)$", re.I)
_DOSE_ROW_RE = re.compile(
    r"\b(?:Adult|Child|Elderly|Infant|Neonate)\s*:?[^\n]{0,140}?\b\d+(?:[.,]\d+)?\s*"
    r"(?:mg|g|micrograms?|mcg|units?|mL|millilitres?)\b", re.I
)
_ROUTE_LINE_RE = re.compile(r"^(?:BY|TO|INTO?|VIA|ON)\b", re.I)
_PREPARATION_END_RE = re.compile(
    r"\b(?:TABLETS?|CAPSULES?|SUPPOSITORIES|ENEMAS?|INJECTIONS?|"
    r"INFUSIONS?|ORAL\s+(?:SOLUTION|SUSPENSION)|EYE\s+DROPS|"
    r"POWDER\s+FOR\s+(?:ORAL\s+)?SOLUTION)\b",
    re.I,
)


def split_label(text: str) -> tuple[str | None, str]:
    """('PREGNANCY', 'Avoid unless...') for a block that starts with a BNF label."""
    m = _LABEL_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end():].lstrip(" \n:-\u2013")


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
@dataclass
class LBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float = 0.0
    bold: bool = False
    color: int = 0
    zone: str = "body"          # header | footer | side | body
    kind: str = "text"          # text | table
    page_index: int = 0
    dose_box: bool = False
    table_title: str | None = None
    table_id: str | None = None
    drug_hint: str | None = None


def is_blue(color: int) -> bool:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return b >= 110 and b > r + 45


def _dose_box_rects(page, blocks: list[LBlock]) -> list:
    """Find pale-blue dose panels, including unlabeled adjacent panels.

    The fill alone is not enough: another BNF callout can use it. A panel must
    either carry the printed field label, or contain both a route and a
    population-specific dose row.
    """
    target = (224, 231, 245)  # BNF #E0E7F5
    width, height = page.rect.width, page.rect.height
    candidates = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if not fill or len(fill) < 3:
            continue
        rgb = tuple(round(float(channel) * 255) for channel in fill[:3])
        if any(abs(actual - expected) > 3 for actual, expected in zip(rgb, target)):
            continue
        rect = pymupdf.Rect(drawing.get("rect"))
        if rect.width < width * 0.22 or rect.height < 30 or rect.width * rect.height < width * height * 0.01:
            continue
        inside = [
            block for block in blocks
            if rect.contains(pymupdf.Point((block.x0 + block.x1) / 2, (block.y0 + block.y1) / 2))
        ]
        content = "\n".join(block.text for block in inside)
        has_label = bool(re.search(r"\bINDICATIONS\s+AND\s+DOSE\b", content, re.I))
        has_route = bool(re.search(r"\b(?:BY|TO|INTO?|VIA|ON)\s+(?:[A-Z]+\s*){1,5}", content, re.I))
        has_dose = bool(_DOSE_ROW_RE.search(content))
        if has_label or (has_route and has_dose):
            candidates.append(rect)

    candidates.sort(key=lambda rect: rect.width * rect.height, reverse=True)
    unique = []
    for rect in candidates:
        area = rect.width * rect.height
        for kept in unique:
            overlap = rect & kept
            intersection = max(0.0, overlap.width) * max(0.0, overlap.height)
            smaller = min(area, kept.width * kept.height)
            if smaller and intersection / smaller >= 0.92:
                break
        else:
            unique.append(rect)
    return unique


def extract_blocks(page, header_frac: float = 0.04, footer_frac: float = 0.06) -> list[LBlock]:
    """Extract one styled block per PDF text line and mark layout zones/panels."""
    width, height = page.rect.width, page.rect.height
    out: list[LBlock] = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            text = "".join(span["text"] for span in spans)
            if not text.strip():
                continue
            sizes, bolds, colors = Counter(), Counter(), Counter()
            for span in spans:
                count = len(span["text"].strip())
                if count:
                    sizes[round(span["size"], 1)] += count
                    bolds[bool(span["flags"] & 16)] += count
                    colors[span["color"]] += count
            x0, y0, x1, y1 = line.get("bbox", block["bbox"])
            direction = line.get("dir", (1, 0))
            rotated = (round(direction[0]), round(direction[1])) != (1, 0)
            if rotated or x0 > width * 0.93 or x1 < width * 0.07:
                zone = "side"
            elif y0 < height * header_frac:
                zone = "header"
            elif y0 > height * (1 - footer_frac):
                zone = "footer"
            else:
                zone = "body"
            out.append(LBlock(
                x0, y0, x1, y1, text,
                sizes.most_common(1)[0][0] if sizes else 0.0,
                bolds.most_common(1)[0][0] if bolds else False,
                colors.most_common(1)[0][0] if colors else 0,
                zone,
            ))
    for rect in _dose_box_rects(page, out):
        for block in out:
            center = pymupdf.Point((block.x0 + block.x1) / 2, (block.y0 + block.y1) / 2)
            if rect.contains(center):
                block.dose_box = True
    font_counts = Counter()
    for block in out:
        if block.size > 0:
            font_counts[round(block.size, 1)] += len(block.text.strip())
    body_size = font_counts.most_common(1)[0][0] if font_counts else 9.0
    drug_titles = [
        b for b in out
        if b.zone == "body" and is_blue(b.color) and b.size >= body_size * 1.2
        and not b.text.strip().isupper() and 2 < len(b.text.strip()) < 220
    ]
    for block in (b for b in out if b.dose_box):
        center_x, center_y = (block.x0 + block.x1) / 2, (block.y0 + block.y1) / 2
        nearest = min(
            drug_titles,
            key=lambda title: abs((title.x0 + title.x1) / 2 - center_x)
            + abs((title.y0 + title.y1) / 2 - center_y) * 0.65,
            default=None,
        )
        if nearest:
            block.drug_hint = nearest.text.strip()
    return out


def shaded_bnf_tables(page, blocks: list[LBlock]) -> list[tuple[object, str, str, str]]:
    """Recover BNF list tables drawn as adjacent blue title and pale body bands.

    PyMuPDF's line-table detector misses the Appendix 1 drug lists because they
    have no cell rules. These regions have a stronger BNF-specific signature:
    a blue filled band containing a blue ``TABLE n`` label and title, followed
    immediately by a pale-blue rectangle containing aligned list columns.
    Dose panels are excluded because they have no ``TABLE n`` title band.
    """
    width = page.rect.width
    header_fills, body_fills, compact_fills, blue_rules = [], [], [], []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if not fill or len(fill) < 3:
            continue
        rgb = tuple(round(float(channel) * 255) for channel in fill[:3])
        rect = pymupdf.Rect(drawing.get("rect"))
        if all(abs(a - b) <= 8 for a, b in zip(rgb, (0, 113, 188))) and rect.height <= 2:
            blue_rules.append(rect)
            continue
        if rect.width < width * 0.6 or rect.height < 12:
            if (rect.width >= width * 0.25 and rect.height >= 30
                    and all(abs(a - b) <= 8 for a, b in zip(rgb, (232, 237, 247)))):
                compact_fills.append(rect)
            continue
        if all(abs(a - b) <= 8 for a, b in zip(rgb, (199, 213, 238))):
            header_fills.append(rect)
        elif all(abs(a - b) <= 8 for a, b in zip(rgb, (225, 232, 245))):
            body_fills.append(rect)

    found = []
    for header in header_fills:
        title_lines = [
            b for b in blocks
            if header.contains(pymupdf.Point((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2))
        ]
        marker = next((b for b in title_lines if re.fullmatch(r"TABLE\s+\d+", b.text.strip(), re.I)), None)
        if marker is None or not is_blue(marker.color):
            continue
        body = [
            rect for rect in body_fills
            if abs(rect.x0 - header.x0) <= 2
            and abs(rect.x1 - header.x1) <= 2
            and abs(rect.y0 - header.y1) <= 2
        ]
        if not body:
            continue
        body_rect = max(body, key=lambda rect: rect.height)
        table_rect = header | body_rect
        title = next(
            (b.text.strip() for b in sorted(title_lines, key=lambda b: b.y0)
             if b is not marker and is_blue(b.color) and b.text.strip()),
            "",
        )
        content = [
            b for b in blocks
            if body_rect.contains(pymupdf.Point((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2))
        ]
        # These BNF appendix tables are lists laid out in independent columns,
        # not row-wise relationships. Preserve each column top-to-bottom.
        columns: list[list[LBlock]] = []
        for block in sorted(content, key=lambda b: (b.x0, b.y0)):
            column = next((items for items in columns if abs(items[0].x0 - block.x0) <= 12), None)
            if column is None:
                columns.append([block])
            else:
                column.append(block)
        columns.sort(key=lambda items: items[0].x0)
        column_items = [[b.text.strip() for b in sorted(items, key=lambda b: b.y0) if b.text.strip()]
                        for items in columns]
        n_rows = max((len(items) for items in column_items), default=0)
        table_id = marker.text.strip().split()[-1]
        lines = [f"{marker.text.strip()}: {title}".rstrip()]
        notes = [b.text.strip() for b in sorted(title_lines, key=lambda b: b.y0)
                 if b is not marker and b.text.strip() != title]
        if notes:
            lines.extend(notes)
        if column_items:
            headers = [f"Drug list column {i + 1}" for i in range(len(column_items))]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in range(n_rows):
                cells = [items[row] if row < len(items) else "" for items in column_items]
                lines.append("| " + " | ".join(cells) + " |")
        found.append((table_rect, "\n".join(lines), title, table_id))

    # Compact BNF comparison tables (for example, iron salt/iron-content) use
    # a title above a pale grid with blue horizontal rules rather than a
    # ``TABLE n`` band. Require the title, multiple columns, and repeated rules
    # together so ordinary dose panels and callouts do not match.
    for rect in compact_fills:
        if any(not (rect & prior[0]).is_empty for prior in found):
            continue
        rules = [
            rule for rule in blue_rules
            if rect.x0 - 1 <= rule.x0 and rule.x1 <= rect.x1 + 1
            and rect.y0 - 1 <= (rule.y0 + rule.y1) / 2 <= rect.y1 + 1
            and rule.width >= rect.width * 0.65
        ]
        if len({round((rule.y0 + rule.y1) / 2, 1) for rule in rules}) < 3:
            continue
        title_block = min(
            (b for b in blocks
             if b.x0 < rect.x1 and b.x1 > rect.x0 and 0 <= rect.y0 - b.y1 <= 8
             and len(b.text.strip()) < 120 and b.size >= 6.8),
            key=lambda b: rect.y0 - b.y1,
            default=None,
        )
        if title_block is None:
            continue
        cells = [
            b for b in blocks
            if rect.contains(pymupdf.Point((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2))
        ]
        rows: list[list[LBlock]] = []
        for block in sorted(cells, key=lambda b: (b.y0, b.x0)):
            row = next((items for items in rows if abs(items[0].y0 - block.y0) <= 2), None)
            if row is None:
                rows.append([block])
            else:
                row.append(block)
        rows = [sorted(row, key=lambda b: b.x0) for row in rows]
        if len(rows) < 3 or max((len(row) for row in rows), default=0) < 2:
            continue
        title = title_block.text.strip()
        lines = [title, "| " + " | ".join(b.text.strip() for b in rows[0]) + " |",
                 "| " + " | ".join("---" for _ in rows[0]) + " |"]
        lines.extend("| " + " | ".join(b.text.strip() for b in row) + " |" for row in rows[1:])
        table_rect = rect | pymupdf.Rect(title_block.x0, title_block.y0, title_block.x1, title_block.y1)
        found.append((table_rect, "\n".join(lines), title, ""))
    return found



def estimate_body_size(doc, sample: int = 40) -> float:
    """Dominant font size by character count: the body text size."""
    counts: Counter = Counter()
    n = doc.page_count
    for i in range(0, n, max(1, n // sample)):
        for b in doc.load_page(i).get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for s in ln["spans"]:
                    counts[round(s["size"], 1)] += len(s["text"].strip())
    return counts.most_common(1)[0][0] if counts else 9.0


# --------------------------------------------------------------------------
# Reading order
# --------------------------------------------------------------------------
def find_gutter(blocks: list[LBlock], page_w: float) -> float | None:
    """Centre of the widest vertical white strip in the middle of the page.

    Wide blocks are ignored so a full-width heading cannot hide the gutter, and
    the search band excludes the margins, so the side tab cannot be mistaken
    for a column.
    """
    narrow = [b for b in blocks if (b.x1 - b.x0) < page_w * 0.6]
    free = [x for x in range(int(page_w * 0.35), int(page_w * 0.65))
            if not any(b.x0 <= x <= b.x1 for b in narrow)]
    if not free:
        return None
    runs, cur = [], [free[0]]
    for x in free[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    runs.append(cur)
    best = max(runs, key=len)
    return (best[0] + best[-1]) / 2 if len(best) >= 8 else None


def order_blocks(blocks: list[LBlock], page_w: float) -> list[LBlock]:
    """Right column then left column, with full-width blocks as band separators.
    Inside a column the order is exact (y0, x0): no bucketing."""
    key = lambda b: (b.y0, b.x0)  # noqa: E731
    g = find_gutter(blocks, page_w)
    if g is None:
        return sorted(blocks, key=key)
    spanners = sorted((b for b in blocks if b.x0 < g - 8 and b.x1 > g + 8), key=key)
    spanner_ids = {id(b) for b in spanners}
    rest = [b for b in blocks if id(b) not in spanner_ids]
    out: list[LBlock] = []
    start = float("-inf")
    for sp in spanners + [None]:
        stop = sp.y0 if sp else float("inf")
        band = [b for b in rest if start <= b.y0 < stop]
        out += sorted((b for b in band if b.x0 >= g), key=key)
        out += sorted((b for b in band if b.x0 < g), key=key)
        if sp:
            out.append(sp)
        start = stop
    return out


# --------------------------------------------------------------------------
# Page furniture: printed page number, running title, chapter tab
# --------------------------------------------------------------------------
def printed_page(header: list[LBlock]) -> str | None:
    """The page number printed in the header ('1264', 'xii'), not the PDF index."""
    for b in sorted(header, key=lambda b: b.x0):
        toks = b.text.split()
        if not toks or toks[0].upper() == "BNF":
            continue
        has_bnf = any(t.upper() == "BNF" for t in toks)
        for t in (toks[0], toks[-1]):
            if _PAGE_TOKEN_RE.fullmatch(t):
                return t
            if _ROMAN_RE.fullmatch(t) and (len(toks) == 1 or has_bnf):
                return t.lower()
    return None


def running_title(header: list[LBlock], page_no: str | None) -> str | None:
    t = " ".join(b.text.replace("\n", " ") for b in header)
    t = _BNF_TAG_RE.sub("", t)
    if page_no:
        t = re.sub(rf"\b{re.escape(page_no)}\b", "", t, count=1)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def chapter_tab(side: list[LBlock]) -> tuple[str | None, str | None]:
    """(chapter_no, chapter_name) from the rotated side tab and its number box."""
    name = no = None
    for b in side:
        t = re.sub(r"\s+", " ", b.text.replace("\n", " ")).strip()
        if re.fullmatch(r"\d{1,2}", t):
            no = t
        elif len(t) > 3 and not re.fullmatch(r"[\d\W]+", t):
            name = t
    return no, name


# --------------------------------------------------------------------------
# Tagging: the hierarchy, tracked across pages
# --------------------------------------------------------------------------
@dataclass
class Ctx:
    chapter_no: str | None = None
    chapter: str | None = None
    subchapter: str | None = None            # running header
    section: str | None = None               # "7 Hypothalamic ..."
    subsection: str | None = None            # "7.1 Adrenocortical ..."
    subsubsection: str | None = None         # "6.3a Chronic hepatitis B"
    classification: str | None = None        # "ANTIVIRALS > NUCLEOSIDE ANALOGUES"
    drug: str | None = None
    preparation: str | None = None       # product/form label inside a dose panel
    review_date: str | None = None
    field: str | None = None
    indication: str | None = None            # bold line inside INDICATIONS AND DOSE
    route: str | None = None                 # "BY MOUTH", "BY INTRAVENOUS INJECTION"
    freq: str | None = None                  # side-effect frequency band
    printed_page: str | None = None
    table_title: str | None = None
    table_id: str | None = None
    interaction_source: str | None = None
    interaction_target: str | None = None
    interaction_id: str | None = None

    def label(self) -> str:
        parts = [self.chapter, self.section, self.subsection, self.subsubsection, self.drug]
        return " > ".join(p for p in parts if p)


@dataclass
class Tagged:
    text: str
    kind: str                 # text | forms | alert | table | heading | title
    page: int                 # 1-based PDF page
    ctx: Ctx
    dose_box: bool = False


class BnfTagger:
    def __init__(self, body_size: float, title_ratio: float = 1.25, heading_ratio: float = 1.3):
        self.body = body_size
        self.title_min = body_size * title_ratio
        self.heading_min = body_size * heading_ratio
        self.c = Ctx()
        self.in_forms = False
        self._pending_date: str | None = None
        self._carry: str | None = None   # indication heading waiting to join its first dose line
        self._dose_started = False
        self._interaction_counter = 0

    # -- per page ---------------------------------------------------------
    def feed(self, page_index: int, blocks: list[LBlock], page_w: float) -> list[Tagged]:
        header = [b for b in blocks if b.zone == "header"]
        side = [b for b in blocks if b.zone == "side"]
        body = [b for b in blocks if b.zone == "body"]

        pp = printed_page(header)
        self.c.printed_page = pp
        title = running_title(header, pp)
        if title:
            # Numbered headings can be carried for many pages. A changed
            # running title marks a new BNF topic; clear the numbered
            # subheading and drug/field context that belonged to the prior
            # topic before accepting the new title. Without this, e.g. the
            # "3.2 Myasthenia gravis" heading leaks into the later musculoskeletal
            # chapter's Naproxen chunks.
            if self.c.subchapter and title != self.c.subchapter:
                self._reset_below("chapter")
            self.c.subchapter = title
        no, name = chapter_tab(side)
        if name and name != self.c.chapter:
            self.c.chapter, self.c.chapter_no = name, no
            self._reset_below("chapter")
        if name and re.search(r"appendix\s*1.*interaction|interaction.*appendix\s*1", name, re.I):
            if title and not re.search(r"appendix\s*1\s+interactions?", title, re.I):
                self.c.field = "INTERACTIONS"

        out: list[Tagged] = []
        ordered = order_blocks(body, page_w)
        gutter = find_gutter(body, page_w)
        if self.c.field == "INDICATIONS AND DOSE" and ordered and not ordered[0].dose_box:
            # A dose field is scoped to its shaded panel. Do not carry it into
            # unrelated page-top prose when the next page begins outside that
            # panel; an unlabelled continuation panel will re-establish it.
            self.c.field = self.c.indication = self.c.route = None
            self._dose_started = False
        right_indices = [i for i, block in enumerate(ordered) if gutter is not None and block.x0 >= gutter]
        last_right_index = max(right_indices, default=None)
        right_context = None
        for i, b in enumerate(ordered):
            out += self._tag(b, page_index + 1)
            if i == last_right_index:
                right_context = self._snap()
        # The next BNF page resumes at its right column. Preserve the context
        # at the end of this page's right-column stream; the left column has
        # already been emitted, but its last monograph must not bleed into the
        # next page's continuation.
        if right_context is not None:
            self.c = right_context
        return out

    # -- per block --------------------------------------------------------
    def _snap(self) -> Ctx:
        return replace(self.c)

    def _reset_below(self, level: str) -> None:
        order = ["chapter", "section", "subsection", "subsubsection", "drug"]
        for name in order[order.index(level) + 1:]:
            setattr(self.c, name, None)
        self.c.field = self.c.classification = self.c.review_date = None
        self.c.preparation = None
        self.c.indication = self.c.route = self.c.freq = None
        self.c.interaction_source = self.c.interaction_target = self.c.interaction_id = None
        self._carry = None
        self._dose_started = False
        self.in_forms = False

    def _tag(self, b: LBlock, page: int) -> list[Tagged]:
        text = _PUA_RE.sub("", b.text)
        text = _LEAD_JUNK_RE.sub("", text).strip()
        if not text:
            return []
        flat = re.sub(r"\s*\n\s*", " ", text)

        # Form/product labels printed inside the pale dose panel (for example
        # "ASACOL MR 400MG TABLETS") identify the preparation used by following
        # indication and dose rows. Retain this context in the dose chunk's
        # searchable metadata instead of leaving the label stranded in prose.
        if not b.dose_box and self.c.preparation:
            self.c.preparation = None

        if b.kind == "table":
            ctx = self._snap()
            ctx.field = "TABLE"
            ctx.table_title = b.table_title
            ctx.table_id = b.table_id
            return [Tagged(text, "table", page, ctx, False)]

        # A dose panel may continue at the top of the next column/page without
        # repeating its blue field label. Its fill, route, population-dose row,
        # and nearest monograph title supply the missing local context.
        if b.dose_box and self.c.field != "INDICATIONS AND DOSE":
            self.c.field = "INDICATIONS AND DOSE"
            self.c.indication = self.c.route = None
            if b.drug_hint:
                self.c.drug = b.drug_hint

        # review date printed beside a drug title
        if re.fullmatch(_DATE_RE.pattern, flat):
            if self.c.drug and not self.c.review_date:
                self.c.review_date = flat
            else:
                self._pending_date = flat
            return []

        # numbered section headings: "7", "7.1", "6.3a"
        m = _NUM_HEADING_RE.match(flat)
        if m and b.size >= self.heading_min and len(flat) < 160:
            num, sub, letter, title = m.groups()
            label = f"{num}{'.' + sub if sub else ''}{letter or ''} {title}"
            if letter:
                self._reset_below("subsection"); self.c.subsubsection = label
            elif sub:
                self._reset_below("section"); self.c.subsection = label
            else:
                self._reset_below("chapter"); self.c.section = label
            return [Tagged(flat, "heading", page, self._snap(), b.dose_box)]

        # classification path: blue, upper-case, with a chevron
        if flat.isupper() and re.search(r"[\u203a>]", flat) and len(flat) < 120:
            self.c.classification = re.sub(r"\s*[\u203a>]\s*", " > ", flat)
            return []

        label, rest = split_label(text)
        if label:
            self.c.field = label
            self.c.preparation = None
            self.c.indication = self.c.route = self.c.freq = None
            self.c.interaction_source = self.c.interaction_target = self.c.interaction_id = None
            self._carry = None
            self._dose_started = False
            self.in_forms = label == "MEDICINAL FORMS"
            if not rest:
                return []
            kind = "forms" if self.in_forms else "alert" if label == "IMPORTANT SAFETY INFORMATION" else "text"
            return self._emit_body(rest, kind, page, b, structure=False)

        if (
            b.dose_box
            and self.c.field == "INDICATIONS AND DOSE"
            and flat.isupper()
            and _PREPARATION_END_RE.search(flat)
            and len(flat) < 180
        ):
            self.c.preparation = flat.strip(" :-")
            self.c.indication = self.c.route = None
            self._carry = None
            self._dose_started = False
            return []

        # drug / class-monograph title: blue and clearly larger than body text
        if is_blue(b.color) and b.size >= self.title_min and len(flat) < 220:
            date = _DATE_RE.search(flat)
            self._reset_below("subsubsection")
            self.c.drug = _DATE_RE.sub("", flat).strip(" -")
            self.c.review_date = date.group(1) if date else self._pending_date
            self._pending_date = None
            return [Tagged(self.c.drug, "title", page, self._snap(), b.dose_box)]

        kind = "forms" if self.in_forms else "text"
        return self._emit_body(text, kind, page, b)

    @staticmethod
    def _looks_like_indication(text: str) -> bool:
        """Recognize short, unpunctuated dose-indication captions."""
        return (
            2 < len(text) < 220
            and text[:1].isupper()
            and not text.isupper()
            and not _POP_RE.match(text)
            and not re.search(r"[.!?;:]\s*$", text)
        )

    def _emit_body(self, text: str, kind: str, page: int, b: LBlock, structure: bool = True) -> list[Tagged]:
        """Body text, with dose indication/route/frequency tracking."""
        flat = re.sub(r"\s*\n\s*", " ", text)
        if self.c.field == "INTERACTIONS":
            self.c.interaction_source = self.c.interaction_target = None
            relation = bool(re.search(
                r"\b(?:is|are)\s+predicted\s+to\b|\b(?:might|may|can)\s+"
                r"(?:slightly\s+|markedly\s+)?(?:increase|decrease|alter)\b|"
                r"\b(?:increases|decreases|inhibit|induces?)\b",
                flat,
                re.I,
            ))
            pair = re.search(
                r"(?P<source>[^.;:]{2,100}?)\s+(?:is|are)\s+predicted\s+to\s+"
                r"(?:increase|decrease|inhibit|induce)[^.;]{0,100}?\b(?:to|with)\s+"
                r"(?P<target>[A-Za-z][A-Za-z0-9()'’ -]{1,60})",
                flat,
                re.I,
            )
            if pair:
                self.c.interaction_source = pair.group("source").strip()
                self.c.interaction_target = pair.group("target").strip().rstrip(" ,")
            if relation:
                if pair:
                    self.c.interaction_id = (
                        f"{self.c.interaction_source.casefold()}->{self.c.interaction_target.casefold()}"
                    )
                else:
                    self._interaction_counter += 1
                    self.c.interaction_id = f"p{page}:i{self._interaction_counter}"
            elif flat[:1].isupper() and flat.strip().casefold() not in {
                "study", "theoretical", "moderate", "severe", "avoid", "monitor", "adjust dose"
            }:
                # Standalone bold drug names mark the next alphabetic entry.
                self.c.interaction_id = None
        if structure and self.c.field == "INDICATIONS AND DOSE":
            if flat.isupper() and _ROUTE_LINE_RE.match(flat):
                self.c.route = flat
                return []
            if flat.isupper() and self.c.route and _ROUTE_TAIL_RE.fullmatch(flat):
                self.c.route = f"{self.c.route} {flat}"
                return []
            if _POP_RE.match(flat):
                self._dose_started = True
            elif self._looks_like_indication(flat):
                if self._dose_started:
                    self.c.indication = flat
                    self.c.route = None
                    self._dose_started = False
                    self._carry = flat
                    return []
                if not self.c.route:
                    self.c.indication = f"{self.c.indication} {flat}".strip() if self.c.indication else flat
                    self._carry = self.c.indication
                    return []
        if self._carry and kind == "text":
            if (
                self.c.field == "INDICATIONS AND DOSE"
                and not self.c.route
                and not _POP_RE.match(flat)
                and not flat.isupper()
            ):
                self.c.indication = f"{self.c.indication} {flat}".strip()
                self._carry = self.c.indication
                return []
            text, self._carry = f"{self._carry} {text}", None
        if self.c.field == "SIDE-EFFECTS" and kind == "text":
            out, pos, freq = [], 0, self.c.freq
            for match in _FREQ_RE.finditer(text):
                if text[pos:match.start()].strip():
                    self.c.freq = freq
                    out.append(Tagged(text[pos:match.start()].strip(), kind, page, self._snap(), b.dose_box))
                freq = self.c.freq = match.group(1)
                pos = match.end()
            if text[pos:].strip():
                self.c.freq = freq
                out.append(Tagged(text[pos:].strip(), kind, page, self._snap(), b.dose_box))
            return out
        return [Tagged(text, kind, page, self._snap(), b.dose_box)]



# --------------------------------------------------------------------------
# Stitching across column / page breaks
# --------------------------------------------------------------------------
_END_RE = re.compile(r"[.!?:;)\]]\s*$")


def _continues(prev: Tagged, nxt: Tagged) -> bool:
    if prev.kind != "text" or nxt.kind != "text":
        return False
    if (prev.ctx.drug, prev.ctx.field) != (nxt.ctx.drug, nxt.ctx.field):
        return False
    if (
        prev.ctx.interaction_id != nxt.ctx.interaction_id
        or prev.ctx.interaction_source != nxt.ctx.interaction_source
        or prev.ctx.interaction_target != nxt.ctx.interaction_target
        or prev.ctx.preparation != nxt.ctx.preparation
    ):
        return False
    if prev.dose_box != nxt.dose_box:
        return False
    first = nxt.text.lstrip()[:1]
    return bool(first) and first.islower() and not _END_RE.search(prev.text)


def stitch(items: list[Tagged]) -> list[Tagged]:
    out: list[Tagged] = []
    for it in items:
        if out and _continues(out[-1], it):
            out[-1] = Tagged(out[-1].text.rstrip() + " " + it.text.lstrip(),
                             out[-1].kind, out[-1].page, out[-1].ctx, out[-1].dose_box)
        else:
            out.append(it)
    return out


# --------------------------------------------------------------------------
# Tables: drop only text that is mostly inside a table box
# --------------------------------------------------------------------------
def mostly_inside(b: LBlock, box, frac: float = 0.6) -> bool:
    r = pymupdf.Rect(b.x0, b.y0, b.x1, b.y1)
    inter = r & box
    return (not inter.is_empty) and inter.get_area() >= frac * max(r.get_area(), 1e-6)


def merge_tables(blocks: list[LBlock], tables: list[tuple]) -> list[LBlock]:
    """Replace source text in detected regions with one structural table block."""
    normalized = [(entry[0], entry[1], *(entry[2:] or (None, None))) for entry in tables]
    keep = [b for b in blocks if not any(mostly_inside(b, box) for box, _, _, _ in normalized)]
    for box, md, title, table_id in normalized:
        keep.append(LBlock(box.x0, box.y0, box.x1, box.y1, md, kind="table",
                           table_title=title, table_id=table_id))
    return keep


# --------------------------------------------------------------------------
# Glue for PDFParser
# --------------------------------------------------------------------------
def ctx_meta(t: Tagged) -> dict:
    """Flat, JSON-safe hierarchy and dose provenance for chunks."""
    c = t.ctx
    meta = {
        "kind": t.kind, "chapter_no": c.chapter_no, "chapter": c.chapter,
        "subchapter": c.subchapter, "section": c.section, "subsection": c.subsection,
        "subsubsection": c.subsubsection, "classification": c.classification,
        "drug": c.drug, "review_date": c.review_date, "field": c.field,
        "indication": c.indication, "route": c.route, "freq": c.freq,
        "printed_page": c.printed_page, "pdf_page": t.page,
        "table_title": c.table_title, "table_id": c.table_id,
        "interaction_source": c.interaction_source,
        "interaction_target": c.interaction_target,
        "interaction_id": c.interaction_id,
        "h1": c.chapter, "h2": c.section or c.subsection, "h3": c.subsubsection,
        "entity": c.drug,
        "preparation": c.preparation,
        "entity_type": "drug" if c.drug else ("topic" if c.label() else None),
        "dose_box": t.dose_box,
        "dose_box_color": "#E0E7F5" if t.dose_box else None,
    }
    return {key: value for key, value in meta.items() if value}



def bnf_pages(doc, page_indices, *, tables_fn=None, body_size: float | None = None) -> list[Tagged]:
    """Run layout tagging sequentially so monograph context carries across pages.

    tables_fn(page) -> [(Rect, markdown)] supplies candidate tables. A candidate
    dominated by dose-panel lines is discarded: BNF dose boxes are structures,
    not tables, even when PyMuPDF detects aligned text rows inside them.
    """
    effective_body_size = body_size or estimate_body_size(doc)
    tagger = BnfTagger(effective_body_size)
    items: list[Tagged] = []
    previous_index = None
    for index in page_indices:
        if previous_index is not None and index != previous_index + 1:
            # Sampled, non-contiguous pages must not inherit a drug/field from
            # the prior sample. Full-book ingestion keeps sequential context.
            tagger = BnfTagger(effective_body_size)
        previous_index = index
        page = doc.load_page(index)
        blocks = extract_blocks(page)
        tables = shaded_bnf_tables(page, blocks)
        if tables_fn:
            usable_tables = []
            for box, markdown in tables_fn(page):
                rect = pymupdf.Rect(box)
                inside = [
                    block for block in blocks
                    if rect.contains(pymupdf.Point((block.x0 + block.x1) / 2, (block.y0 + block.y1) / 2))
                ]
                dose_lines = sum(block.dose_box for block in inside)
                if len(inside) >= 3 and dose_lines / len(inside) >= 0.8:
                    continue
                usable_tables.append((rect, markdown))
            tables.extend(usable_tables)
        if tables:
            blocks = merge_tables(blocks, tables)
        items.extend(tagger.feed(index, blocks, page.rect.width))
    return stitch(items)
