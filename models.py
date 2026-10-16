import base64
import hashlib
import json
import time
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import config

def check_key():
    if not config.GEMINI_API_KEY:
        raise SystemExit("No GEMINI_API_KEY found. Copy .env.example to .env and add your key.")


def get_chat_model(slow=False):
    check_key()
    # slow=True adds a rate limiter so the free tier does not block us
    limiter = None
    if slow:
        limiter = InMemoryRateLimiter(requests_per_second=config.REQUESTS_PER_SECOND, max_bucket_size=1)
    return ChatGoogleGenerativeAI(
        model=config.GEN_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        max_retries=config.MAX_RETRIES,
        rate_limiter=limiter,
    )


class GeminiEmbeddings(Embeddings):
    def __init__(self):
        check_key()
        self.model = GoogleGenerativeAIEmbeddings(
            model=config.EMBED_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            output_dimensionality=config.EMBED_DIM,
        )

    def retry(self, function):
        # wait and try again when the free tier says "too many requests"
        wait = 5
        for attempt in range(config.MAX_RETRIES):
            try:
                return function()
            except Exception as error:
                if not any(word in str(error) for word in ("429", "RESOURCE_EXHAUSTED", "503")):
                    raise
                print(f"rate limit, waiting {wait} seconds")
                time.sleep(wait)
                wait = min(wait * 2, 60)
        return function()

    def embed_documents(self, texts):
        vectors = []
        # small batches with a pause between them
        for start in range(0, len(texts), config.EMBED_BATCH):
            batch = texts[start:start + config.EMBED_BATCH]
            vectors += self.retry(lambda: self.model.embed_documents(batch))
            time.sleep(config.EMBED_DELAY)
        return vectors

    def embed_query(self, text):
        return self.retry(lambda: self.model.embed_query(text))


embeddings = None


def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = GeminiEmbeddings()
    return embeddings


vision_chain = None


def describe_image(png_bytes, kind, caption, table_text):
    if kind == "table":
        task = "Describe this TABLE: what it is about, the column and row headers, and the key values."
    else:
        task = "Describe this FIGURE: its type, title, axes, labels, the main trend and the key numbers."
    prompt = f"{task} Plain text, at most 150 words, so it can be found by search.\nCaption: {caption or 'none'}"
    if table_text:
        prompt += "\nExtracted table text (may have mistakes):\n" + table_text[:3000]

    # the same image + prompt is only sent to Gemini once
    key = hashlib.sha1(config.GEN_MODEL.encode() + prompt.encode() + png_bytes).hexdigest()
    cache = {}
    if config.CACHE_FILE.exists():
        cache = json.loads(config.CACHE_FILE.read_text(encoding="utf-8"))
    if key in cache:
        return cache[key]

    global vision_chain
    if vision_chain is None:
        vision_chain = get_chat_model(slow=True) | StrOutputParser()

    picture = base64.b64encode(png_bytes).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + picture}},
    ])
    text = " ".join(vision_chain.invoke([message]).split())

    if text:
        cache[key] = text
        config.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return text
