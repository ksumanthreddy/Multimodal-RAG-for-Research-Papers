import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
import db
import models
from pdf_extractor import extract_pdf

splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=100)


def make_chunks(blocks):
    # blocks are (text, can_split), tables and figures are never split
    chunks = []
    current = ""
    for text, can_split in blocks:
        pieces = [text]
        if can_split and len(text) > config.CHUNK_SIZE:
            pieces = splitter.split_text(text)
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > config.CHUNK_SIZE:
                chunks.append(current)
                current = ""
            current = current + "\n\n" + piece if current else piece
    if current:
        chunks.append(current)
    return chunks


def short_path(path):
    # save image paths relative to the project folder
    try:
        return str(Path(path).resolve().relative_to(config.BASE_DIR))
    except ValueError:
        return str(path)


def ingest_pdf(pdf_path, dry_run=False):
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.name
    print("reading", pdf_name)
    pages = extract_pdf(pdf_path, config.CROPS_DIR / pdf_path.stem)
    if dry_run:
        return

    page_rows = []
    asset_rows = []
    texts = []
    metadatas = []
    ids = []

    for page in pages:
        page_parts = []
        search_blocks = []

        for item in page.ordered_items():
            if item["kind"] == "text":
                page_parts.append(item["text"])
                search_blocks.append((item["text"], True))
                continue

            visual = item["visual"]
            kind = visual["content_type"]
            key = f"{pdf_name}_p{page.page_no}_{visual['content_id']}"
            print("describing", key)

            # Gemini writes a summary of the figure or table
            png = Path(visual["image_path"]).read_bytes()
            summary = models.describe_image(png, kind, visual["caption"], visual["markdown"])

            table_md = None
            if kind == "table":
                table_md = visual["markdown"][:config.MAX_TABLE_CHARS]
            asset_rows.append((key, pdf_name, page.page_no, kind, visual["caption"],
                               summary, table_md, short_path(visual["image_path"])))

            # the page keeps a placeholder, the search text keeps the summary
            tag = "TBL" if kind == "table" else "FIG"
            page_parts.append(f"[[{tag}:{key}]]")
            search_text = f"{kind.upper()} {visual['content_id']} on page {page.page_no}."
            search_text += f"\nCaption: {visual['caption'] or ''}\nDescription: {summary}"
            if table_md:
                search_text += f"\nTable data:\n{table_md}"
            search_blocks.append((search_text, False))

        page_rows.append((page.page_no, "\n\n".join(page_parts)))

        for number, content in enumerate(make_chunks(search_blocks), start=1):
            texts.append(content)
            metadatas.append({"pdf_name": pdf_name, "page_no": page.page_no})
            ids.append(f"{pdf_name}::p{page.page_no}::c{number}")

    print("embedding", len(texts), "chunks")
    vectors = models.get_embeddings().embed_documents(texts)

    # old data is only deleted after Gemini finished
    print("saving")
    db.delete_pdf(pdf_name)
    for page_no, page_text in page_rows:
        db.save_page(pdf_name, page_no, page_text)
    for row in asset_rows:
        db.save_asset(row)
    if texts:
        texts = [text.replace("\x00", "") for text in texts]
        db.get_vector_store().add_embeddings(texts=texts, embeddings=vectors, metadatas=metadatas, ids=ids)
    print(f"done: {len(page_rows)} pages, {len(asset_rows)} figures/tables, {len(texts)} chunks")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv
    pdf_files = [Path(a) for a in args] or sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found. Put PDFs in {config.PDF_DIR}")
    if not dry_run:
        db.create_tables()
    for pdf_file in pdf_files:
        ingest_pdf(pdf_file, dry_run)
