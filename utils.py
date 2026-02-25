import base64
import json
import re
from pathlib import Path
from PIL import Image
import io

from config import MAX_IMAGE_DIMENSION, MAX_IMAGE_SIZE_MB, SUPPORTED_FORMATS


def is_supported_image(path: Path) -> bool:
    """Check if file is a supported image format."""
    return path.suffix.lower() in SUPPORTED_FORMATS


def prepare_image_for_api(image_path: Path) -> tuple[str, str]:
    """
    Load, resize if needed, and encode image as base64.
    Returns (base64_data, media_type).
    """
    with Image.open(image_path) as img:
        # Convert RGBA or palette images to RGB for JPEG encoding
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Resize if image exceeds the Claude Vision dimension limit
        max_dim = max(img.width, img.height)
        if max_dim > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max_dim
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # Encode to JPEG bytes in memory
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

        # If still too large, re-encode at lower quality
        size_mb = buffer.getbuffer().nbytes / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            buffer.seek(0)

        b64 = base64.standard_b64encode(buffer.read()).decode("utf-8")
        return b64, "image/jpeg"


def collect_images(photo_dir: Path) -> list[Path]:
    """Recursively collect all supported image files from a directory."""
    images = []
    for path in sorted(photo_dir.rglob("*")):
        if path.is_file() and is_supported_image(path):
            images.append(path)
    return images


def clean_ocr_text(text: str) -> str:
    """Normalize whitespace in OCR output while preserving Chinese characters."""
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).strip()
