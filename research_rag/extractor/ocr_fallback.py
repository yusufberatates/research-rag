"""OCR fallback for scanned pages.

Used only when a page's embedded text layer is too short to be real text
(i.e. the page is likely a scanned image). Renders the page to a raster
image with pymupdf and feeds it to pytesseract -- no external readers or
file-execution involved.
"""
from __future__ import annotations

import fitz  # pymupdf

from research_rag.config import OCR_DPI

try:
    import pytesseract
    from PIL import Image
    import io

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


def ocr_available() -> bool:
    return _OCR_AVAILABLE


def ocr_page(page: "fitz.Page") -> str:
    """Render a single PDF page and run OCR on it. Returns extracted text,
    or empty string if pytesseract/PIL are not installed."""
    if not _OCR_AVAILABLE:
        return ""
    zoom = OCR_DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError:
        return ""
