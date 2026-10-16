import base64
import pymupdf
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import config
import db
import models


# 1. Database
try:
    print("1) PostgreSQL:", db.run("SELECT version()")[0][0].split(",")[0])
    db.run("CREATE EXTENSION IF NOT EXISTS vector")
    version = db.run("SELECT extversion FROM pg_extension WHERE extname = 'vector'")[0][0]
    print("   pgvector version:", version)
except Exception as e:
    raise SystemExit(f"Database error: {e}")


# 2. Gemini key
try:
    models.check_key()
    print("2) Gemini key found")
except Exception as e:
    raise SystemExit(f"Gemini key error: {e}")


# 3. Text model
chat = models.get_chat_model() | StrOutputParser()

try:
    print("3) Text model:", chat.invoke("Reply with exactly one word: ready")[:40])
except Exception as e:
    raise SystemExit(
        f"Text model failed: {e}\n"
        "Check the model name or try again later if it is a 503 error."
    )

# 4. Vision model
try:
    image = pymupdf.Pixmap(
        pymupdf.csRGB,
        pymupdf.IRect(0, 0, 96, 96),
        False
    )
    image.set_rect(image.irect, (255, 0, 0))

    image_data = base64.b64encode(
        image.tobytes("png")
    ).decode("utf-8")

    message = HumanMessage(content=[
        {"type": "text", "text": "What single color fills this image? One word."},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + image_data
            }
        }
    ])

    print("4) Vision:", chat.invoke([message])[:40])

except Exception as e:
    raise SystemExit(f"Vision test failed: {e}")


# 5. Embeddings
try:
    vector = models.get_embeddings().embed_query("hello world")
    print(f"5) Embeddings: {len(vector)} numbers")

    if len(vector) != config.EMBED_DIM:
        print(f"Warning: expected {config.EMBED_DIM} dimensions")

except Exception as e:
    raise SystemExit(f"Embedding test failed: {e}")


print("\nAll checks passed.")