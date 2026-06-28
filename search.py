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
import threading
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

# e5 の生コサインは無関係な日本語同士でも高め（0.75前後）に出る。
# 「関連あり」と見なす最低ライン。これ未満は検索結果から除外する。
MIN_SCORE = float(os.environ.get("CASE_FINDER_MIN_SCORE", "0.80"))
# 表示用の「関連度(0-100%)」へ変換する際の下限・上限（この区間を0〜100%に伸縮）。
REL_FLOOR = float(os.environ.get("CASE_FINDER_REL_FLOOR", "0.78"))
REL_CEIL = float(os.environ.get("CASE_FINDER_REL_CEIL", "0.92"))
# グラフのエッジ（事例同士の線）を引く最低類似度。
EDGE_FLOOR = float(os.environ.get("CASE_FINDER_EDGE_FLOOR", "0.82"))


# ──────────────────────────────────────────────────────────────
# 埋め込みモデル（遅延ロード）
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _model():
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


def relevance(cos: float) -> float:
    """生コサインを直感的な関連度 0.0〜1.0 に変換する。"""
    if REL_CEIL <= REL_FLOOR:
        return max(0.0, min(1.0, cos))
    return max(0.0, min(1.0, (cos - REL_FLOOR) / (REL_CEIL - REL_FLOOR)))


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
            industry  TEXT NOT NULL DEFAULT '',
            problem   TEXT NOT NULL DEFAULT '',
            action    TEXT NOT NULL DEFAULT '',
            result    TEXT NOT NULL DEFAULT '',
            embedding BLOB NOT NULL
        )
        """
    )
    # 既存DB向けの簡易マイグレーション（不足列を追加）
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(cases)")}
    for col in ("industry", "problem", "action", "result"):
        if col not in existing:
            conn.execute(f"ALTER TABLE cases ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
    conn.commit()


def upsert_case(conn, title, source, text, excerpt, industry, fields: dict, embedding: np.ndarray):
    conn.execute(
        """
        INSERT INTO cases (title, source, text, excerpt, industry, problem, action, result, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source) DO UPDATE SET
            title=excluded.title, text=excluded.text, excerpt=excluded.excerpt,
            industry=excluded.industry, problem=excluded.problem,
            action=excluded.action, result=excluded.result, embedding=excluded.embedding
        """,
        (
            title, source, text, excerpt, industry,
            fields.get("problem", ""), fields.get("action", ""), fields.get("result", ""),
            embedding.astype(np.float32).tobytes(),
        ),
    )
    conn.commit()


def delete_sources(conn, sources):
    if not sources:
        return
    conn.executemany("DELETE FROM cases WHERE source = ?", [(s,) for s in sources])
    conn.commit()


def all_sources(conn):
    return {r["source"] for r in conn.execute("SELECT source FROM cases")}


def load_all(conn):
    """全事例を (メタ情報リスト, ベクトル行列) で返す。"""
    rows = conn.execute(
        "SELECT id, title, source, text, excerpt, industry, problem, action, result, embedding FROM cases"
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
                "industry": r["industry"],
                "problem": r["problem"],
                "action": r["action"],
                "result": r["result"],
            }
        )
        mats.append(np.frombuffer(r["embedding"], dtype=np.float32))
    matrix = np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32)
    return meta, matrix


# ──────────────────────────────────────────────────────────────
# インメモリ・インデックス（毎クエリのSQLite全読込を回避）
# ──────────────────────────────────────────────────────────────
_LOCK = threading.Lock()
_INDEX = None  # {"meta":..., "matrix":..., "mtime":...}


def _db_mtime():
    return os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0.0


def get_index():
    """事例メタとベクトル行列をメモリから返す。DB更新は mtime で自動検知。"""
    global _INDEX
    with _LOCK:
        mt = _db_mtime()
        if _INDEX is None or _INDEX["mtime"] != mt:
            conn = connect()
            init_db(conn)
            meta, matrix = load_all(conn)
            conn.close()
            _INDEX = {"meta": meta, "matrix": matrix, "mtime": _db_mtime()}
        return _INDEX["meta"], _INDEX["matrix"]


def invalidate_cache():
    """ingest/upload 後に明示的にキャッシュを破棄する。"""
    global _INDEX
    with _LOCK:
        _INDEX = None


# ──────────────────────────────────────────────────────────────
# 検索 + グラフ
# ──────────────────────────────────────────────────────────────
def search(query: str, top_k: int = 6, industry: str = "", min_score: float | None = None):
    """クエリに意味的に近い事例を返す。ベクトルは正規化済みなので内積=コサイン類似度。"""
    if min_score is None:
        min_score = MIN_SCORE
    meta, matrix = get_index()
    if not meta or not query.strip():
        return {"nodes": [], "edges": []}

    qv = embed([query], "query")[0]
    scores = matrix @ qv  # (N,)

    nodes = []
    for i in np.argsort(-scores):
        s = float(scores[i])
        if s < min_score:
            break  # 降順なので以降も閾値未満
        m = meta[i]
        if industry and m["industry"] != industry:
            continue
        nodes.append(
            {
                "id": m["id"],
                "title": m["title"],
                "source": m["source"],
                "excerpt": m["excerpt"],
                "industry": m["industry"],
                "problem": m["problem"],
                "action": m["action"],
                "result": m["result"],
                "score": s,
                "relevance": relevance(s),
                "_idx": int(i),
            }
        )
        if len(nodes) >= top_k:
            break

    edges = _edges(nodes, matrix)
    for n in nodes:
        n.pop("_idx", None)
    return {"nodes": nodes, "edges": edges}


def _edges(nodes, matrix):
    """事例同士の関連（グラフの細い線）。閾値は分位点と下限の大きい方で動的に決める。"""
    pairs = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            sim = float(matrix[nodes[a]["_idx"]] @ matrix[nodes[b]["_idx"]])
            pairs.append((nodes[a]["id"], nodes[b]["id"], sim))
    if not pairs:
        return []
    sims = np.array([p[2] for p in pairs])
    thr = max(EDGE_FLOOR, float(np.percentile(sims, 70)))
    return [{"a": a, "b": b, "sim": s} for (a, b, s) in pairs if s >= thr]


def list_industries():
    meta, _ = get_index()
    return sorted({m["industry"] for m in meta if m["industry"]})


def stats():
    meta, _ = get_index()
    return {"count": len(meta), "model": MODEL_NAME, "industries": list_industries()}
