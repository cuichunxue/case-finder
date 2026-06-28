"""data/ 内の PPT・PDF を読み取り、意味ベクトル化して cases.db に登録する。

使い方:
    python ingest.py            # data/ 全体を取り込み
    python ingest.py foo.pdf    # 個別ファイルを取り込み

1ファイル = 1事例 として扱います。タイトルはファイル名（拡張子なし）です。
"""

from __future__ import annotations

import os
import sys

import ocr
import search
from extract import classify_industry, extract_fields
from search import DATA_DIR, connect, embed, init_db, upsert_case

SUPPORTED = (".pdf", ".pptx", ".ppt", ".txt", ".md")

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


def ingest_file(conn, path: str, ocr_ok: bool, industry: str | None = None) -> bool:
    title = os.path.splitext(os.path.basename(path))[0]
    source = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
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
    # タイトルを先頭に足して意味の手掛かりを強める
    vec = embed([f"{title}\n{text}"], "passage")[0]
    upsert_case(conn, title, source, text, make_excerpt(text), ind, fields, vec)
    got = [name for name, key in (("課題", "problem"), ("施策", "action"), ("成果", "result")) if fields.get(key)]
    print(f"  ✓ {source}  ({len(text)} 文字 / 業種: {ind or '不明'} / 抽出: {('・'.join(got)) or 'なし'})")
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
        on_disk = {
            os.path.relpath(p, os.path.dirname(os.path.abspath(__file__)))
            for p in iter_targets([])
        }
        stale = search.all_sources(conn) - on_disk
        if stale:
            search.delete_sources(conn, stale)
            print(f"  ・削除済みファイルのレコードを {len(stale)} 件掃除しました")

    conn.close()
    search.invalidate_cache()
    print(f"\n完了: {ok}/{len(targets)} 件を登録しました。`python app.py` で起動できます。")


if __name__ == "__main__":
    main(sys.argv[1:])
