import time
import anthropic
from pathlib import Path
from rich.console import Console

from config import ANTHROPIC_API_KEY, OCR_MODEL, API_RATE_LIMIT_DELAY
from utils import prepare_image_for_api, clean_ocr_text

console = Console()

# Prompt optimized for mixed Chinese/English document OCR
OCR_SYSTEM_PROMPT = """You are a precise OCR assistant. Extract ALL text from the image exactly as it appears.
Rules:
- Output ONLY the extracted text, no explanations or commentary
- Preserve original line breaks and paragraph structure
- Keep Chinese characters exactly as written
- Keep mixed Chinese/English text in its original order
- If there is no text in the image, output exactly: [NO TEXT]
- Do not translate, summarize, or paraphrase"""

OCR_USER_PROMPT = "Extract all text from this image."


class OCRProcessor:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Copy .env.example to .env and add your key."
            )
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._last_call_time = 0.0

    def _rate_limit(self):
        """Enforce minimum delay between API calls."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < API_RATE_LIMIT_DELAY:
            time.sleep(API_RATE_LIMIT_DELAY - elapsed)
        self._last_call_time = time.monotonic()

    def extract_text(self, image_path: Path) -> dict:
        """
        Extract text from a single image using Claude Vision.
        Returns dict with keys: text, tokens_used, error
        """
        result = {"text": "", "tokens_used": 0, "error": None}

        try:
            b64_data, media_type = prepare_image_for_api(image_path)

            self._rate_limit()

            response = self.client.messages.create(
                model=OCR_MODEL,
                max_tokens=4096,
                system=OCR_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": OCR_USER_PROMPT,
                            },
                        ],
                    }
                ],
            )

            raw_text = response.content[0].text
            result["tokens_used"] = (
                response.usage.input_tokens + response.usage.output_tokens
            )

            # Skip images with no text
            if raw_text.strip() == "[NO TEXT]":
                result["text"] = ""
            else:
                result["text"] = clean_ocr_text(raw_text)

        except anthropic.RateLimitError:
            console.print("[yellow]Rate limit hit, waiting 60s...[/yellow]")
            time.sleep(60)
            return self.extract_text(image_path)  # Retry once

        except anthropic.APIError as e:
            result["error"] = f"API error: {str(e)}"
            console.print(f"[red]API error for {image_path.name}: {e}[/red]")

        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            console.print(f"[red]Error processing {image_path.name}: {e}[/red]")

        return result
