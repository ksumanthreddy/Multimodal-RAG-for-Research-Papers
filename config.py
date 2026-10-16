import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# free tier limits
REQUESTS_PER_SECOND = float(os.getenv("REQUESTS_PER_SECOND", "0.25"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))
EMBED_DELAY = float(os.getenv("EMBED_DELAY", "1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))

# PostgreSQL (LangChain wants the postgresql+psycopg form)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pdf_rag")
LANGCHAIN_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
COLLECTION = "pdf_chunks"

# folders
PDF_DIR = BASE_DIR / "data" / "pdfs"
CROPS_DIR = BASE_DIR / "storage" / "crops"
CACHE_FILE = BASE_DIR / "storage" / "vision_cache.json"

# settings used by pdf_extractor.py
IMAGE_DPI = 200
CROP_PADDING = 4
MIN_IMAGE_WIDTH = 40
MIN_IMAGE_HEIGHT = 40
CAPTION_TOLERANCE = 30
DETECT_VECTOR_FIGURES = True
VECTOR_FIGURE_PADDING = 12
HEADER_MAX_FRACTION = 0.02
FOOTER_MIN_FRACTION = 0.98

# chunking and search
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
MAX_TABLE_CHARS = 6000
SEARCH_K = int(os.getenv("SEARCH_K", "8"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "3"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "4"))
