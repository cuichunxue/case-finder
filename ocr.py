"""画像ベースの PPT/PDF から文字を読み取る OCR モジュール（Tesseract 日本語）。

生成AI（LLM）は使いません。テキスト層を持たないスキャンPDFや、
文字が画像化されたスライドから文字を取り出すために使います。

必要なもの（OSパッケージ）:
    macOS : brew install tesseract tesseract-lang
    Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-jpn
そのうえで: pip install pytesseract Pillow
"""

from __future__ import annotations

import io
import os

OCR_LANG = os.environ.get("CASE_FINDER_OCR_LANG", "jpn+eng")
OCR_DPI = int(os.environ.get("CASE_FINDER_OCR_DPI", "220"))


def available() -> bool:
    """Tesseract 本体が使えるか。未導入なら OCR をスキップする判断に使う。"""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def ocr_image_bytes(img_bytes: bytes, lang: str = OCR_LANG) -> str:
    from PIL import Image
    import pytesseract

    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img, lang=lang).strip()


def ocr_pdf_page(page, lang: str = OCR_LANG, dpi: int = OCR_DPI) -> str:
    """PyMuPDF のページを画像化して OCR する。"""
    pix = page.get_pixmap(dpi=dpi)
    return ocr_image_bytes(pix.tobytes("png"), lang=lang)
