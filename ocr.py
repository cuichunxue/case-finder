"""画像ベースの PPT/PDF から文字を読み取る OCR モジュール（生成AI不使用）。

テキスト層を持たないスキャンPDFや、文字が画像化されたスライドから
文字を取り出すために使います。エンジンは環境変数で選べます:

    CASE_FINDER_OCR_ENGINE = easyocr (既定) | tesseract | auto

- EasyOCR  : 日本語精度が高い。Apache-2.0 で商用利用可。pip のみで導入可（モデルは初回DL）。
                pip install easyocr
- Tesseract: 軽量。OS本体の導入が必要。
                macOS : brew install tesseract tesseract-lang
                Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-jpn
                pip   : pip install pytesseract Pillow

どちらも未導入なら OCR は自動でスキップされ、テキスト層のみ取り込みます。
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

ENGINE = os.environ.get("CASE_FINDER_OCR_ENGINE", "easyocr").lower()
OCR_DPI = int(os.environ.get("CASE_FINDER_OCR_DPI", "220"))
# この信頼度未満のOCR結果は捨てる（誤認識テキストの索引混入を防ぐ）
OCR_MIN_CONF = float(os.environ.get("CASE_FINDER_OCR_MIN_CONF", "0.4"))
# EasyOCR の言語（日本語＋英語）。Tesseract は "jpn+eng"。
EASYOCR_LANGS = os.environ.get("CASE_FINDER_OCR_LANG_EASYOCR", "ja,en").split(",")
TESSERACT_LANG = os.environ.get("CASE_FINDER_OCR_LANG_TESSERACT", "jpn+eng")


# ──────────────────────────────────────────────────────────────
# EasyOCR
# ──────────────────────────────────────────────────────────────
def _easyocr_installed() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr

    import device

    # GPUが使えれば利用（CASE_FINDER_DEVICE）。モデルは初回のみ自動ダウンロード。
    return easyocr.Reader(EASYOCR_LANGS, gpu=device.use_gpu(), verbose=False)


def _filter_ocr(results, min_conf: float) -> str:
    """EasyOCR の (bbox, text, conf) 列から低信頼を除外して連結（純関数・テスト用）。"""
    return "\n".join(
        t for (_b, t, c) in results if c >= min_conf and str(t).strip()
    ).strip()


def _easyocr_image(img_bytes: bytes) -> str:
    import numpy as np
    from PIL import Image

    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    # detail=1 で信頼度を取得し、しきい値で誤認識を除外
    results = _easyocr_reader().readtext(img, detail=1, paragraph=False)
    return _filter_ocr(results, OCR_MIN_CONF)


# ──────────────────────────────────────────────────────────────
# Tesseract
# ──────────────────────────────────────────────────────────────
def _tesseract_installed() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def _tesseract_image(img_bytes: bytes) -> str:
    from PIL import Image
    import pytesseract

    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img, lang=TESSERACT_LANG).strip()


# ──────────────────────────────────────────────────────────────
# エンジン選択
# ──────────────────────────────────────────────────────────────
def _active_engine() -> str | None:
    """実際に使えるエンジン名を返す。使えなければ None。"""
    if ENGINE == "easyocr":
        return "easyocr" if _easyocr_installed() else None
    if ENGINE == "tesseract":
        return "tesseract" if _tesseract_installed() else None
    # auto: EasyOCR を優先
    if _easyocr_installed():
        return "easyocr"
    if _tesseract_installed():
        return "tesseract"
    return None


def available() -> bool:
    return _active_engine() is not None


def engine_name() -> str:
    return _active_engine() or "なし"


def ocr_image_bytes(img_bytes: bytes) -> str:
    eng = _active_engine()
    if eng == "easyocr":
        return _easyocr_image(img_bytes)
    if eng == "tesseract":
        return _tesseract_image(img_bytes)
    return ""


def ocr_pdf_page(page) -> str:
    """PyMuPDF のページを画像化して OCR する。"""
    pix = page.get_pixmap(dpi=OCR_DPI)
    return ocr_image_bytes(pix.tobytes("png"))
