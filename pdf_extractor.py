from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import config

try:
    import camelot
    HAS_CAMELOT = True
except Exception:
    camelot = None
    HAS_CAMELOT = False

def overlap_ratio(candidate, reference):
    intersection = candidate & reference
    if intersection.is_empty:
        return 0

    candidate_area = candidate.width * candidate.height
    if candidate_area == 0:
        return 0
    overlap_area = intersection.width * intersection.height
    return overlap_area / candidate_area


def overlaps_content(candidate, boxes, threshold=0.50):
    for content_rect in boxes:
        ratio = overlap_ratio(candidate, content_rect)
        if ratio > threshold:
            return True
    return False


def build_checkers(fraud_rects, logo_rects, header_footer_zones):

    def overlaps_fraud(candidate, threshold=0.50):
        return overlaps_content(candidate, fraud_rects, threshold)

    def matches_logo(candidate, threshold=0.50):
        return overlaps_content(candidate, logo_rects, threshold)

    def overlaps_header_footer(candidate):
        for zone in header_footer_zones:
            if overlap_ratio(candidate, zone) > 0:
                return True
        return False

    return overlaps_fraud, matches_logo, overlaps_header_footer

def duplicates(PDF_PATH):
    doc = pymupdf.open(PDF_PATH)
    n = len(doc)

    if n < 10:
        page_count = min(n, max(2, n // 2)) if n >= 2 else n
    elif n <= 14:
        page_count = 5
    elif n <= 20:
        page_count = 7
    else:
        page_count = 10

    page_counts = list(range(page_count))
    bbox_occurrences = defaultdict(list)

    for page_num in page_counts:
        page = doc[page_num]

        for block in page.get_text("dict")["blocks"]:
            bbox = tuple(float(v) for v in block["bbox"])
            bbox_occurrences[bbox].append({
                "page": page_num + 1,
                "type": "text" if block.get("type") == 0 else "image"
            })

        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect:
                bbox = tuple(float(v) for v in rect)
                bbox_occurrences[bbox].append({
                    "page": page_num + 1,
                    "type": "drawing"
                })

    fraud_box = []
    logo_box = []

    for bbox, occurrences in bbox_occurrences.items():
        pages = sorted(set(item["page"] for item in occurrences))
        if len(pages) < 2:
            continue

        x0, y0, x1, y1 = bbox
        bbox_area = max(0, x1 - x0) * max(0, y1 - y0)
        large_pages = set()

        for page_num in pages:
            page_area = doc[page_num - 1].rect.width * doc[page_num - 1].rect.height
            if page_area > 0 and bbox_area / page_area > 0.10:
                large_pages.add(page_num)

        is_image = any(item["type"] == "image" for item in occurrences)

        if is_image and len(large_pages) >= 5:
            logo_box.append(bbox)
        else:
            fraud_box.append(bbox)

    doc.close()
    return fraud_box, logo_box

def header_footer(page, fraud_boxes):
    HEADER_FOOTER_MARGIN = 5
    page_width = page.rect.width
    page_height = page.rect.height

    header_candidates = []
    footer_candidates = []

    for bbox in fraud_boxes:
        x0, y0, x1, y1 = bbox

        if y0 < page_height * config.HEADER_MAX_FRACTION:
            header_candidates.append(bbox)

        if y1 > page_height * config.FOOTER_MIN_FRACTION:
            footer_candidates.append(bbox)

    header_zone = None
    footer_zone = None

    if header_candidates:
        header_bottom = max(bbox[3] for bbox in header_candidates)
        header_zone = pymupdf.Rect(
            0, 0, page_width,
            min(page_height, header_bottom + HEADER_FOOTER_MARGIN)
        )

    if footer_candidates:
        footer_top = min(bbox[1] for bbox in footer_candidates)
        footer_zone = pymupdf.Rect(
            0, max(0, footer_top - HEADER_FOOTER_MARGIN),
            page_width, page_height
        )

    # bottom 5% is always excluded
    footer_zone_5pct = pymupdf.Rect(0, page_height * 0.95, page_width, page_height)

    if footer_zone is None:
        footer_zone = footer_zone_5pct
    else:
        footer_zone |= footer_zone_5pct

    return [z for z in [header_zone, footer_zone] if z is not None]


def rows_to_markdown(rows):
    rows = [[("" if c is None else " ".join(str(c).split())).replace("|", "/")
             for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def extract_tables(PDF_PATH, PAGE_NUMBER, page, page_height,
                   overlaps_header_footer, overlaps_fraud):
    table_boxes = []
    table_markdown = []

    try:
        pymupdf_tables = page.find_tables()
        pymupdf_has_tables = len(pymupdf_tables.tables) > 0
    except Exception:
        pymupdf_has_tables = False

    if pymupdf_has_tables:
        if not HAS_CAMELOT:
            print(f"   [warn] page {PAGE_NUMBER} has tables but camelot is not installed")
            return table_boxes, table_markdown
        try:
            tables = camelot.read_pdf(
                PDF_PATH, pages=str(PAGE_NUMBER), flavor="lattice"
            )
            if len(tables) == 0:
                tables = camelot.read_pdf(
                    PDF_PATH, pages=str(PAGE_NUMBER), flavor="stream"
                )

            for table in tables:
                x0, y0, x1, y1 = table._bbox

                # Camelot bottom-left -> PyMuPDF top-left
                rect = pymupdf.Rect(x0, page_height - y1, x1, page_height - y0)

                if overlaps_header_footer(rect):
                    continue
                if overlaps_fraud(rect):
                    continue

                table_boxes.append(rect)
                table_markdown.append(rows_to_markdown(table.df.values.tolist()))
        except Exception as e:
            print(f"   [warn] Camelot failed on page {PAGE_NUMBER}: {e}")

    return table_boxes, table_markdown

def extract_images(page, overlaps_header_footer, overlaps_fraud, matches_logo):
    image_boxes = []
    try:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 1:
                continue

            rect = pymupdf.Rect(block["bbox"])

            # icons, bullets, thin lines
            if rect.width < config.MIN_IMAGE_WIDTH or rect.height < config.MIN_IMAGE_HEIGHT:
                continue
            if overlaps_header_footer(rect):
                continue
            if overlaps_fraud(rect):
                continue
            if matches_logo(rect):
                continue

            image_boxes.append(rect)
    except Exception as e:
        print(f"   [warn] image extraction failed: {e}")

    return image_boxes


_CAPTION_FIGURE = re.compile(r"^(?i:figure|fig|image|chart)\.?\s*(?:\d+|[IVX]+\b)")
_CAPTION_TABLE = re.compile(r"^(?i:table|tab)\.?\s*(?:\d+|[IVX]+\b)")


def detect_vector_figures(page, existing, text_records,
                          overlaps_header_footer, overlaps_fraud):
    try:
        clusters = page.cluster_drawings()
    except Exception:
        return []

    page_area = page.rect.width * page.rect.height
    pad = config.VECTOR_FIGURE_PADDING
    boxes = []

    for c in clusters:
        r = pymupdf.Rect(c)
        if r.width < 80 or r.height < 60:
            continue                      # table rules, underlines, small decorations
        if r.width * r.height > 0.85 * page_area:
            continue                      # page border / background

        padded = pymupdf.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad) & page.rect

        # axis labels / legends are text, not drawings: pull in small text blocks touching the chart
        for rec in text_records:
            tb = pymupdf.Rect(rec["bbox"])
            is_caption = bool(_CAPTION_FIGURE.match(rec["text"]) or _CAPTION_TABLE.match(rec["text"]))
            if (not is_caption and tb.width < page.rect.width * 0.5
                    and overlap_ratio(tb, padded) > 0.01):
                padded |= tb
        padded &= page.rect

        if overlaps_header_footer(padded):
            continue
        if overlaps_fraud(padded):
            continue
        # already found as a table or image
        if overlaps_content(padded, existing, 0.30):
            continue
        if any(overlap_ratio(e, padded) > 0.50 for e in existing):
            continue

        boxes.append(padded)
        existing = existing + [padded]
        
    return boxes


def extract_text(page, PAGE_NUMBER, overlaps_header_footer, overlaps_fraud,
                 overlaps_content, table_boxes, image_boxes):
    text_boxes = []
    text_records = []

    try:
        blocks = page.get_text("dict")["blocks"]
        for block_index, block in enumerate(blocks, start=1):

            if block["type"] != 0:
                continue

            rect = pymupdf.Rect(block["bbox"])
            block_text = "\n".join(
                "".join(span["text"] for span in line.get("spans", []))
                for line in block.get("lines", [])
            ).strip()

            if not block_text:
                continue

            if overlaps_header_footer(rect):
                continue
            if overlaps_fraud(rect):
                continue
            if overlaps_content(rect, table_boxes):
                continue
            if overlaps_content(rect, image_boxes):
                continue

            # Do not filter text using logo_boxes.
            # Preserve genuine text beneath watermarks.

            text_boxes.append(rect)
            text_records.append({
                "text_id": f"P{PAGE_NUMBER}_TXT{block_index}",
                "page": PAGE_NUMBER,
                "bbox": tuple(rect),
                "text": block_text,
                "order": block_index,
            })
    except Exception as e:
        print(f"   [warn] text extraction failed: {e}")

    return text_boxes, text_records



def associate_captions(text_records, table_boxes, table_markdown, image_boxes,
                       vertical_tolerance=None):
    tol = vertical_tolerance if vertical_tolerance is not None else config.CAPTION_TOLERANCE
    items = []

    for i, (rect, md) in enumerate(zip(table_boxes, table_markdown), start=1):
        items.append({
            "content_id": f"TBL{i}", "content_type": "table", "bbox": tuple(rect),
            "caption": None, "caption_bbox": None, "explanatory_text": [],
            "markdown": md,
        })
    for i, rect in enumerate(image_boxes, start=1):
        items.append({
            "content_id": f"FIG{i}", "content_type": "figure", "bbox": tuple(rect),
            "caption": None, "caption_bbox": None, "explanatory_text": [],
            "markdown": "",
        })

    for item in items:
        x0, y0, x1, y1 = item["bbox"]
        candidates = []
        for rec in text_records:
            tx0, ty0, tx1, ty1 = rec["bbox"]
            if max(0, min(x1, tx1) - max(x0, tx0)) == 0:
                continue                  # not above/below the content
            if ty1 <= y0:
                distance, position = y0 - ty1, "above"
            elif ty0 >= y1:
                distance, position = ty0 - y1, "below"
            else:
                distance, position = 0, "overlapping"
            if distance <= tol:
                candidates.append((distance, position, rec))

        pattern = _CAPTION_TABLE if item["content_type"] == "table" else _CAPTION_FIGURE
        for distance, position, rec in sorted(candidates, key=lambda c: c[0]):
            text = rec["text"].strip()
            if pattern.match(text) and item["caption"] is None:
                item["caption"], item["caption_bbox"] = text, rec["bbox"]
            elif position != "overlapping":
                item["explanatory_text"].append(text)
    return items


def find_anchor(visual_bbox, page_blocks):
    rect = pymupdf.Rect(visual_bbox)
    inside = [i for i, b in enumerate(page_blocks, start=1)
              if overlap_ratio(pymupdf.Rect(b["bbox"]), rect) > 0.5]
    if inside:
        return float(min(inside))
    below = [i for i, b in enumerate(page_blocks, start=1) if b["bbox"][1] >= rect.y0]
    return min(below) - 0.5 if below else float(len(page_blocks) + 1)


def crop_region(page, bbox, out_path: Path, dpi: int | None = None) -> Path:
    pad = config.CROP_PADDING
    r = pymupdf.Rect(bbox)
    clip = pymupdf.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad) & page.rect
    pix = page.get_pixmap(clip=clip, dpi=dpi or config.IMAGE_DPI)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return out_path


@dataclass
class PageData:
    page_no: int                                      # 1-based
    width: float
    height: float
    text_blocks: list = field(default_factory=list)   # [{"text_id","page","bbox","text","order"}]
    visuals: list = field(default_factory=list)       # tables + figures

    # every item in `visuals` is a dict with:
    #   content_id, content_type ("table"|"figure"), bbox, caption, caption_bbox,
    #   explanatory_text (list[str]), markdown (tables), image_path, page_no, anchor

    def ordered_items(self) -> list[dict]:
        items = [{"kind": "text", "bbox": b["bbox"], "text": b["text"], "order": b["order"]}
                 for b in self.text_blocks]
        items += [{"kind": v["content_type"], "bbox": v["bbox"], "visual": v, "order": v["anchor"]}
                  for v in self.visuals]
        return sorted(items, key=lambda item: item["order"])

def extract_pdf(pdf_path, crops_dir, verbose=True) -> list[PageData]:
    PDF_PATH = str(pdf_path)
    crops_dir = Path(crops_dir)
    stem = Path(PDF_PATH).stem

    fraud_boxes, logo_boxes = duplicates(PDF_PATH)
    fraud_rects = [pymupdf.Rect(b) for b in fraud_boxes]
    logo_rects = [pymupdf.Rect(b) for b in logo_boxes]

    pages: list[PageData] = []
    with pymupdf.open(PDF_PATH) as doc:
        for page_number, page in enumerate(doc, start=1):
            page_height = page.rect.height

            zones = header_footer(page, fraud_boxes)
            overlaps_fraud, matches_logo, overlaps_header_footer = build_checkers(
                fraud_rects, logo_rects, zones
            )

            table_boxes, table_markdown = extract_tables(
                PDF_PATH, page_number, page, page_height,
                overlaps_header_footer, overlaps_fraud
            )
            image_boxes = extract_images(
                page, overlaps_header_footer, overlaps_fraud, matches_logo
            )
            text_boxes, text_records = extract_text(
                page, page_number, overlaps_header_footer, overlaps_fraud,
                overlaps_content, table_boxes, image_boxes
            )

            # optional: vector charts/diagrams
            if config.DETECT_VECTOR_FIGURES:
                existing = list(table_boxes) + image_boxes
                image_boxes = image_boxes + detect_vector_figures(
                    page, existing, text_records,
                    overlaps_header_footer, overlaps_fraud
                )

            # captions + explanatory text for tables and figures
            visuals = associate_captions(text_records, table_boxes, table_markdown, image_boxes)

            page_blocks = page.get_text("dict")["blocks"]
            for v in visuals:
                out = crops_dir / f"{stem}_p{page_number}_{v['content_id']}.png"
                crop_region(page, v["bbox"], out)
                v["image_path"] = str(out)
                v["page_no"] = page_number
                v["anchor"] = find_anchor(v["bbox"], page_blocks)

            # body text = not a caption already attached, and not inside a table/figure
            visual_rects = [pymupdf.Rect(v["bbox"]) for v in visuals]
            caption_boxes = {tuple(v["caption_bbox"]) for v in visuals if v["caption_bbox"]}
            body = [r for r in text_records
                    if tuple(r["bbox"]) not in caption_boxes
                    and not overlaps_content(pymupdf.Rect(r["bbox"]), visual_rects)]

            pages.append(PageData(page_number, page.rect.width, page_height, body, visuals))

            if verbose:
                n_t = sum(v["content_type"] == "table" for v in visuals)
                n_f = sum(v["content_type"] == "figure" for v in visuals)
                print(f"page {page_number:>3}: {len(body):>3} text | {n_t} table | {n_f} figure")

    return pages
