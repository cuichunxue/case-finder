"""data/ 内の PPT・PDF を読み取り、意味ベクトル化して cases.db に登録する。

使い方:
    python ingest.py            # data/ 全体を取り込み
    python ingest.py foo.pdf    # 個別ファイルを取り込み

1ファイル = 1事例 とし、本文は節単位のチャンクに分割して埋め込みます
（長文・複数トピックの取りこぼしを防ぐため）。
"""

from __future__ import annotations

import os
import re
import sys

import ocr
import search
from extract import classify_industry, extract_fields
from search import DATA_DIR, connect, embed, init_db, store_case

SUPPORTED = (".pdf", ".pptx", ".ppt", ".txt", ".md")

# チャンク分割の目安（文字数）とオーバーラップ（境界での分断を緩和）
CHUNK_SIZE = int(os.environ.get("CASE_FINDER_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.environ.get("CASE_FINDER_CHUNK_OVERLAP", "60"))

# テキスト層がこの文字数未満のページ/スライドは画像中心とみなし OCR にかける
OCR_TRIGGER_CHARS = 12


def extract_pdf(path: str, ocr_ok: bool) -> str:
    import fitz  # PyMuPDF

    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            t = page.get_text().strip()
            if len(t) < OCR_TRIGGER_CHARS and ocr_ok:
                try:
                    t = ocr.ocr_pdf_page(page)
                except Exception:  # noqa: BLE001
                    pass
            if t:
                parts.append(t)
    return "\n".join(parts)


def extract_pptx(path: str, ocr_ok: bool) -> str:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(path)
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs)
                    if line.strip():
                        parts.append(line)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text for c in row.cells]
                    parts.append(" | ".join(cells))
            # 画像化されたスライド/図中の文字を OCR で拾う
            if ocr_ok and shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    parts.append(ocr.ocr_image_bytes(shape.image.blob))
                except Exception:  # noqa: BLE001
                    pass
    return "\n".join(p for p in parts if p.strip())


def extract_text(path: str, ocr_ok: bool = False) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path, ocr_ok)
    if ext in (".pptx", ".ppt"):
        return extract_pptx(path, ocr_ok)
    if ext in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"未対応の形式: {ext}")


def make_excerpt(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """本文を節単位のチャンクに分割する（行/段落でまとめ、長すぎる塊は強制分割）。

    overlap>0 のとき、各チャンク先頭に直前チャンク末尾を少し重ねて、
    文が境界で分断されることによる取りこぼしを緩和する。
    """
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(p) > size:  # 長い段落は固定長で割る
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(p), size):
                chunks.append(p[i:i + size])
        elif len(cur) + len(p) + 1 <= size:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    chunks = chunks or [text[:size]]

    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            out.append((tail + "\n" + chunks[i]).strip())
        chunks = out
    return chunks


def ingest_file(conn, path: str, ocr_ok: bool, industry: str | None = None) -> bool:
    title = os.path.splitext(os.path.basename(path))[0]
    source = search.rel_source(path)
    try:
        text = extract_text(path, ocr_ok).strip()
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {source}: 読み取り失敗 ({e})")
        return False
    if not text:
        hint = "" if ocr_ok else "（画像中心の可能性。OCRを有効にすると読める場合があります）"
        print(f"  ⚠ {source}: テキストを抽出できませんでした{hint}")
        return False

    # BERT埋め込みで「課題/施策/成果」に分類、業種を推定（指定があれば優先）
    fields = extract_fields(text)
    ind = industry if industry is not None else classify_industry(text)
    # 本文をチャンク化し、各チャンクをタイトル付きで埋め込む
    pieces = chunk_text(text)
    vecs = embed([f"{title}\n{c}" for c in pieces], "passage")
    chunks = list(zip(pieces, vecs))
    store_case(conn, title, source, text, make_excerpt(text), ind, fields, chunks)
    got = [name for name, key in (("課題", "problem"), ("施策", "action"), ("成果", "result")) if fields.get(key)]
    print(f"  ✓ {source}  ({len(text)} 文字 / {len(chunks)}チャンク / 業種: {ind or '不明'} / 抽出: {('・'.join(got)) or 'なし'})")
    return True


def iter_targets(args):
    if args:
        for a in args:
            yield a if os.path.isabs(a) else os.path.join(DATA_DIR, a)
        return
    for name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, name)
        if os.path.isfile(path) and name.lower().endswith(SUPPORTED):
            yield path


def main(argv):
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = connect()
    init_db(conn)

    targets = list(iter_targets(argv))
    # 引数指定で対象が無い場合のみ終了。引数なし（全体スキャン）は空でも prune を実行する。
    if not targets and argv:
        print(f"取り込む対象がありません。{DATA_DIR}/ に PPT/PDF を置いてください。")
        conn.close()
        return

    ocr_ok = ocr.available()
    print(f"OCR: {'有効（エンジン: ' + ocr.engine_name() + '）' if ocr_ok else '無効（OCRエンジン未検出。テキスト層のみ取り込み）'}")
    if targets:
        print(f"埋め込みモデルを準備中…（初回のみダウンロード）")
    ok = 0
    for path in targets:
        if ingest_file(conn, path, ocr_ok):
            ok += 1

    # 引数なし（data/全体の取り込み）時は、消えたファイルのDB行を掃除する
    if not argv:
        on_disk = {search.rel_source(p) for p in iter_targets([])}
        stale = search.all_sources(conn) - on_disk
        if stale:
            search.delete_sources(conn, stale)
            print(f"  ・削除済みファイルのレコードを {len(stale)} 件掃除しました")

    conn.close()
    search.invalidate_cache()
    print(f"\n完了: {ok}/{len(targets)} 件を登録しました。`python app.py` で起動できます。")


if __name__ == "__main__":
    main(sys.argv[1:])
