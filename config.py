import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directories ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
SQLITE_DB_PATH = str(DATA_DIR / "photos.db")

# ── Claude API ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OCR_MODEL = "claude-haiku-4-5"
SUMMARIZE_MODEL = "claude-haiku-4-5"

# ── OCR settings ───────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB = 5           # Resize images larger than this
MAX_IMAGE_DIMENSION = 1568      # Max pixels per side (Claude Vision recommended limit)
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
API_RATE_LIMIT_DELAY = 0.5     # Seconds between API calls to avoid rate limiting

# ── Knowledge base ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "photo_ocr_kb"
TOP_K_RESULTS = 5              # Default number of search results

# ── Batch processing ───────────────────────────────────────────────────────
BATCH_SIZE = 10                # Commit to DB every N images
