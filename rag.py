import base64
import re
import sys
from pathlib import Path
from typing import Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
import config
import db
import models

SYSTEM_PROMPT = (
    "You answer questions about PDF documents using ONLY the sources provided. "
    "Sources have normal text, tables as markdown, and the original images of figures. "
    "Read numbers and trends from the images and tables. "
    "Every source starts with a label like [S1]. After each claim, cite it like [S1] or [S2]. "
    "If the sources do not have the answer, say you could not find it. Never invent numbers. Be concise. "
    "If the sources are unrelated to the question, respond exactly: "
    "'I couldn't find relevant information in the indexed PDFs.' "
    "Do not answer using outside knowledge."
)

PLACEHOLDER = re.compile(r"\[\[(FIG|TBL):(.+?)\]\]")


class KeywordRetriever(BaseRetriever):
    k: int = 8
    pdf_name: Optional[str] = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        rows = db.keyword_search(query, self.k, self.pdf_name)
        return [Document(id=row[0], page_content=row[1], metadata=row[2]) for row in rows]


def merge_results(results):
    # reciprocal rank fusion: chunks found by both searches go first
    scores = {}
    docs = {}
    for doc_list in results.values():
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.id or doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            docs[key] = doc
    best_first = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in best_first]


def hybrid_retriever(pdf_name=None):
    search_kwargs = {"k": config.SEARCH_K}
    if pdf_name:
        search_kwargs["filter"] = {"pdf_name": pdf_name}
    vector = db.get_vector_store().as_retriever(search_kwargs=search_kwargs)
    keyword = KeywordRetriever(k=config.SEARCH_K, pdf_name=pdf_name)
    # both searches run on the question, then the results are merged
    return RunnableParallel(vector=vector, keyword=keyword) | RunnableLambda(merge_results)


def text_block(text):
    return {"type": "text", "text": text}


def image_block(path):
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def find_file(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = config.BASE_DIR / path
    return path if path.exists() else None


def asset_blocks(asset, label, pdf_name, page_no, images):
    caption = asset["caption"] or ""
    # tables go as markdown text
    if asset["kind"] == "table":
        return [text_block(f"[Table] {caption}\n{asset['table_md'] or asset['summary'] or ''}")]

    # figures go as the real image until the image limit is reached
    image_file = find_file(asset["image_path"]) if asset["image_path"] else None
    if image_file and len(images) < config.MAX_IMAGES:
        images.append({"label": label, "path": str(image_file), "caption": caption,
                       "pdf_name": pdf_name, "page_no": page_no})
        return [text_block(f"[Figure] {caption} (original image follows)"), image_block(image_file)]
    return [text_block(f"[Figure] {caption}\nDescription: {asset['summary'] or ''}")]


def page_blocks(page_text, label, pdf_name, page_no, images):
    # text stays text, every placeholder becomes the real figure or table
    blocks = []
    position = 0
    for match in PLACEHOLDER.finditer(page_text):
        before = page_text[position:match.start()].strip()
        if before:
            blocks.append(text_block(before))
        position = match.end()
        asset = db.get_asset(match.group(2))
        if asset:
            blocks += asset_blocks(asset, label, pdf_name, page_no, images)
    after = page_text[position:].strip()
    if after:
        blocks.append(text_block(after))
    return blocks


def search(inputs):
    return hybrid_retriever(inputs["pdf_name"]).invoke(inputs["question"])


def choose_pages(inputs):
    # best pages in ranking order, without repeats
    pages = []
    for doc in inputs["docs"]:
        page = (doc.metadata["pdf_name"], doc.metadata["page_no"])
        if page not in pages:
            pages.append(page)
        if len(pages) == config.MAX_PAGES:
            break
    return pages


def build_prompt(inputs):
    blocks = [text_block("Sources:")]
    images = []
    page_list = []
    for number, (pdf_name, page_no) in enumerate(inputs["pages"], start=1):
        page_text = db.get_page_text(pdf_name, page_no)
        if not page_text:
            continue
        label = f"S{number}"
        blocks.append(text_block(f"\n---\n[{label}] (file: {pdf_name}, page {page_no})"))
        blocks += page_blocks(page_text, label, pdf_name, page_no, images)
        page_list.append((label, pdf_name, page_no))
    blocks.append(text_block("\n---\nQuestion: " + inputs["question"]))
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=blocks)]
    return {"messages": messages, "images": images, "page_list": page_list}


def build_chain(model):
    answer_chain = model | StrOutputParser()

    def ask(inputs):
        if not inputs["prompt"]["page_list"]:
            return "I couldn't find relevant information in the indexed PDFs."
        return answer_chain.invoke(inputs["prompt"]["messages"])

    return (
        RunnablePassthrough.assign(docs=RunnableLambda(search))
        | RunnablePassthrough.assign(pages=RunnableLambda(choose_pages))
        | RunnablePassthrough.assign(prompt=RunnableLambda(build_prompt))
        | RunnablePassthrough.assign(answer=RunnableLambda(ask))
    )


chain = None


def cited_numbers(answer):
    numbers = re.findall(r"\[(S\d+)\]", answer)
    return sorted(set(int(n[1:]) for n in numbers))


def answer_question(question, pdf_name=None):
    global chain
    if chain is None:
        chain = build_chain(models.get_chat_model())

    result = chain.invoke({"question": question, "pdf_name": pdf_name})
    answer = result["answer"]
    page_list = result["prompt"]["page_list"]
    images = result["prompt"]["images"]

    # return no sources when relevant information is not found
    if "couldn't find relevant information" in answer.lower() or "could not find" in answer.lower():
        return {"answer": answer, "sources": [], "images": [], "chunks": []}

    # sources come from the database labels, not from the AI memory
    cited = cited_numbers(answer)
    used = [page for page in page_list if int(page[0][1:]) in cited]

    # do not show sources when the answer has no valid citations
    if not cited or not used:
        return {
            "answer": "I couldn't find relevant information in the indexed PDFs.",
            "sources": [],
            "images": [],
            "chunks": []
        }

    sources = [f"{pdf} - page {page_no}" for _label, pdf, page_no in used]
    images = [image for image in images if int(image["label"][1:]) in cited]

    chunks = [{"pdf_name": d.metadata["pdf_name"], "page_no": d.metadata["page_no"],
               "content": d.page_content} for d in result["docs"][:5]]
    return {"answer": answer, "sources": sources, "images": images, "chunks": chunks}


if __name__ == "__main__":
    args = sys.argv[1:]
    pdf_filter = None
    if "--pdf" in args:
        index = args.index("--pdf")
        pdf_filter = args[index + 1]
        del args[index:index + 2]
    if not args:
        raise SystemExit('Usage: python rag.py "your question" [--pdf file.pdf]')

    result = answer_question(" ".join(args), pdf_filter)
    print("\nAnswer:\n" + result["answer"])
    print("\nSources:")
    for line in result["sources"]:
        print("  -", line)
    for image in result["images"]:
        print(f"  image [{image['label']}] {image['pdf_name']} page {image['page_no']}: {image['path']}")
