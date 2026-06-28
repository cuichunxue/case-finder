"""data/ 内の PPT・PDF を読み取り、意味ベクトル化して cases.db に登録する。

使い方:
    python ingest.py            # data/ 全体を取り込み
    python ingest.py foo.pdf    # 個別ファイルを取り込み

1ファイル = 1事例 として扱います。タイトルはファイル名（拡張子なし）です。
"""

from __future__ import annotations

import os
import sys

from search import DATA_DIR, connect, embed, init_db, upsert_case

SUPPORTED = (".pdf", ".pptx", ".ppt", ".txt", ".md")


def extract_pdf(path: str) -> str:
    import fitz  # PyMuPDF

    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


def extract_pptx(path: str) -> str:
    from pptx import Presentation

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
    return "\n".join(parts)


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext in (".pptx", ".ppt"):
        return extract_pptx(path)
    if ext in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"未対応の形式: {ext}")


def make_excerpt(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def ingest_file(conn, path: str) -> bool:
    title = os.path.splitext(os.path.basename(path))[0]
    source = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
    try:
        text = extract_text(path).strip()
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {source}: 読み取り失敗 ({e})")
        return False
    if not text:
        print(f"  ⚠ {source}: テキストを抽出できませんでした（画像PDF等の可能性）")
        return False

    # タイトルを先頭に足して意味の手掛かりを強める
    vec = embed([f"{title}\n{text}"], "passage")[0]
    upsert_case(conn, title, source, text, make_excerpt(text), vec)
    print(f"  ✓ {source}  ({len(text)} 文字)")
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
    if not targets:
        print(f"取り込む対象がありません。{DATA_DIR}/ に PPT/PDF を置いてください。")
        return

    print(f"埋め込みモデルを準備中…（初回のみダウンロード）")
    ok = 0
    for path in targets:
        if ingest_file(conn, path):
            ok += 1
    conn.close()
    print(f"\n完了: {ok}/{len(targets)} 件を登録しました。`python app.py` で起動できます。")


if __name__ == "__main__":
    main(sys.argv[1:])
