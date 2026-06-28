"""事例ファインダーの中核ロジック。

- ローカルの埋め込みモデル（multilingual-e5）でテキストを意味ベクトル化
- SQLite に事例テキストとベクトルを保存
- クエリと事例の意味的な近さ（コサイン類似度）で検索
- 事例同士の関連を計算してグラフ表示に渡す

データは一切外部に送信しません。モデルは初回のみダウンロードし、以後はオフラインで動きます。
"""

from __future__ import annotations

import os
import sqlite3
from functools import lru_cache

import numpy as np

# ──────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cases.db")
DATA_DIR = os.path.join(BASE_DIR, "data")

# 日本語に強く軽量な多言語埋め込みモデル。完全ローカルで動作。
MODEL_NAME = os.environ.get("CASE_FINDER_MODEL", "intfloat/multilingual-e5-small")


# ──────────────────────────────────────────────────────────────
# 埋め込みモデル（遅延ロード）
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _model():
    # import を関数内に置き、モデル未導入でも他の処理が動くようにする
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts, kind: str):
    """テキスト配列を正規化済みベクトルに変換する。

    e5系モデルは用途に応じた接頭辞を付けると精度が上がる:
      - 検索クエリ          -> "query: ..."
      - 事例（被検索文書）  -> "passage: ..."
    """
    assert kind in ("query", "passage")
    prefixed = [f"{kind}: {t}" for t in texts]
    vecs = _model().encode(
        prefixed, normalize_embeddings=True, convert_to_numpy=True
    )
    return vecs.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# SQLite
# ──────────────────────────────────────────────────────────────
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            title     TEXT NOT NULL,
            source    TEXT NOT NULL UNIQUE,
            text      TEXT NOT NULL,
            excerpt   TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()


def upsert_case(conn, title, source, text, excerpt, embedding: np.ndarray):
    conn.execute(
        """
        INSERT INTO cases (title, source, text, excerpt, embedding)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source) DO UPDATE SET
            title=excluded.title, text=excluded.text,
            excerpt=excluded.excerpt, embedding=excluded.embedding
        """,
        (title, source, text, excerpt, embedding.astype(np.float32).tobytes()),
    )
    conn.commit()


def load_all(conn):
    """全事例を (メタ情報リスト, ベクトル行列) で返す。"""
    rows = conn.execute(
        "SELECT id, title, source, text, excerpt, embedding FROM cases"
    ).fetchall()
    meta, mats = [], []
    for r in rows:
        meta.append(
            {
                "id": r["id"],
                "title": r["title"],
                "source": r["source"],
                "text": r["text"],
                "excerpt": r["excerpt"],
            }
        )
        mats.append(np.frombuffer(r["embedding"], dtype=np.float32))
    matrix = np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32)
    return meta, matrix


# ──────────────────────────────────────────────────────────────
# 検索 + グラフ
# ──────────────────────────────────────────────────────────────
def search(query: str, top_k: int = 6, threshold: float = 0.0):
    """クエリに意味的に近い事例を返す。ベクトルは正規化済みなので内積=コサイン類似度。"""
    conn = connect()
    init_db(conn)
    meta, matrix = load_all(conn)
    conn.close()
    if not meta or not query.strip():
        return {"nodes": [], "edges": []}

    qv = embed([query], "query")[0]
    scores = matrix @ qv  # (N,)

    order = np.argsort(-scores)[:top_k]
    nodes = []
    for rank, i in enumerate(order):
        s = float(scores[i])
        if s < threshold:
            continue
        m = meta[i]
        nodes.append(
            {
                "id": m["id"],
                "title": m["title"],
                "source": m["source"],
                "excerpt": m["excerpt"],
                "score": s,
                "_idx": int(i),
            }
        )

    # 事例同士の関連（グラフの細い線）
    edges = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            ia, ib = nodes[a]["_idx"], nodes[b]["_idx"]
            sim = float(matrix[ia] @ matrix[ib])
            if sim >= 0.80:  # 正規化済みe5は全体的に高めに出るため高い閾値
                edges.append(
                    {"a": nodes[a]["id"], "b": nodes[b]["id"], "sim": sim}
                )

    for n in nodes:
        n.pop("_idx", None)
    return {"nodes": nodes, "edges": edges}


def stats():
    conn = connect()
    init_db(conn)
    n = conn.execute("SELECT COUNT(*) AS c FROM cases").fetchone()["c"]
    conn.close()
    return {"count": n, "model": MODEL_NAME}
