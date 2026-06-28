"""事例ファインダーの中核ロジック（生成AI不使用・BERT系/語彙のみ）。

検索の品質を LLM-RAG 同等以上に近づけるため、次を組み合わせる:
  1) チャンク単位の密検索（multilingual-e5 などの BERT 系エンコーダ）
  2) ハイブリッド（BM25 の語彙一致を RRF で融合）→ 固有名詞・数値の取りこぼし対策
  3) クロスエンコーダ・リランカーで上位を並べ替え（最大の精度レバー）
  4) 抜粋ベースの根拠（該当チャンク＋クエリ語ハイライト）→ "なぜ近いか" を提示

いずれも未導入環境では自動でフォールバック（密のみ）し、壊れない。
データは一切外部に送信しない。
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
from functools import lru_cache

import numpy as np

import azure_ai
import rerank

# 埋め込みバックエンド: local（既定・完全ローカル）/ azure（Azure OpenAI 埋め込み）
EMBED_BACKEND = os.environ.get("CASE_FINDER_EMBED_BACKEND", "local").lower()

# ──────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cases.db")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_NAME = os.environ.get("CASE_FINDER_MODEL", "intfloat/multilingual-e5-small")

MIN_SCORE = float(os.environ.get("CASE_FINDER_MIN_SCORE", "0.80"))
WEAK_FLOOR = float(os.environ.get("CASE_FINDER_WEAK_FLOOR", str(max(0.0, MIN_SCORE - 0.08))))
# 密コサイン → 関連度(0-1) の伸縮範囲。下限は WEAK_FLOOR に揃える（弱い帯を表現可能に）。
REL_FLOOR = float(os.environ.get("CASE_FINDER_REL_FLOOR", str(WEAK_FLOOR)))
REL_CEIL = float(os.environ.get("CASE_FINDER_REL_CEIL", "0.92"))
EDGE_FLOOR = float(os.environ.get("CASE_FINDER_EDGE_FLOOR", "0.82"))

# 関連度(0-1)空間での単一の足切り。密/ハイブリッド/リランカーすべてここに集約する。
MIN_REL = float(os.environ.get("CASE_FINDER_MIN_REL", "0.40"))
WEAK_REL = float(os.environ.get("CASE_FINDER_WEAK_REL", "0.15"))
# BM25 生スコアを 0-1 に飽和変換する係数（語彙一致の関連度化）。
BM25_SAT = float(os.environ.get("CASE_FINDER_BM25_SAT", "6.0"))

HYBRID = os.environ.get("CASE_FINDER_HYBRID", "auto")  # auto|off
RRF_K = int(os.environ.get("CASE_FINDER_RRF_K", "60"))
RERANK_TOP = int(os.environ.get("CASE_FINDER_RERANK_TOP", "50"))


# ──────────────────────────────────────────────────────────────
# 埋め込みモデル（遅延ロード）
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts, kind: str):
    assert kind in ("query", "passage")
    # 任意: Azure 埋め込みバックエンド（取り込み・検索で同一バックエンドにすること）
    if EMBED_BACKEND == "azure" and azure_ai.embeddings_available():
        return azure_ai.embed(texts)
    prefixed = [f"{kind}: {t}" for t in texts]
    vecs = _model().encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


def relevance(cos: float) -> float:
    if REL_CEIL <= REL_FLOOR:
        return max(0.0, min(1.0, cos))
    return max(0.0, min(1.0, (cos - REL_FLOOR) / (REL_CEIL - REL_FLOOR)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _lex_rel(bm: float) -> float:
    """BM25 生スコアを 0-1 の関連度へ飽和変換する。"""
    return bm / (bm + BM25_SAT) if bm > 0 else 0.0


def candidate_relevance(c: dict) -> float:
    """候補の統一関連度(0-1)。判断に使った信号を優先する。

      - リランカーが効いた候補 -> sigmoid(リランクスコア)
      - それ以外               -> 密の関連度と語彙(BM25)関連度の大きい方
    これにより「足切り・表示%・並び順」がすべて同じ尺度になる。
    """
    if c.get("rr") is not None:
        return _sigmoid(c["rr"])
    rel = relevance(c["cos"])
    if c.get("bm") is not None:
        rel = max(rel, _lex_rel(c["bm"]))
    return rel


# ──────────────────────────────────────────────────────────────
# トークナイザ（BM25・ハイライト共用。MeCab不要の日本語対応）
# ──────────────────────────────────────────────────────────────
_CJK = r"぀-ヿ一-鿿ｦ-ﾟ"


def tokenize(text: str):
    """英数語＋日本語の文字bigramに分割（語彙一致用）。"""
    text = text.lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-\.]*", text)
    for run in re.findall(f"[{_CJK}]+", text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks += [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


def matched_spans(query: str, text: str, limit: int = 8):
    """クエリ中で text に現れる語句（最大長で重複排除）。ハイライト用。"""
    cands = set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]+", query):
        if len(w) >= 2 and w.lower() in text.lower():
            cands.add(w)
    for run in re.findall(f"[{_CJK}]{{2,}}", query):
        for L in range(min(len(run), 12), 1, -1):
            for i in range(len(run) - L + 1):
                sub = run[i:i + L]
                if sub in text:
                    cands.add(sub)
    maximal = [s for s in cands if not any(s != o and s in o for o in cands)]
    return sorted(maximal, key=len, reverse=True)[:limit]


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
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            title    TEXT NOT NULL,
            source   TEXT NOT NULL UNIQUE,
            text     TEXT NOT NULL,
            excerpt  TEXT NOT NULL,
            industry TEXT NOT NULL DEFAULT '',
            problem  TEXT NOT NULL DEFAULT '',
            action   TEXT NOT NULL DEFAULT '',
            result   TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id   INTEGER NOT NULL,
            ord       INTEGER NOT NULL,
            text      TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_case ON chunks(case_id)")
    conn.commit()


def store_case(conn, title, source, text, excerpt, industry, fields, chunks):
    """事例とそのチャンク群を保存（同一 source は置き換え）。chunks=[(text, vec), ...]。"""
    conn.execute(
        """
        INSERT INTO cases (title, source, text, excerpt, industry, problem, action, result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source) DO UPDATE SET
            title=excluded.title, text=excluded.text, excerpt=excluded.excerpt,
            industry=excluded.industry, problem=excluded.problem,
            action=excluded.action, result=excluded.result
        """,
        (title, source, text, excerpt, industry,
         fields.get("problem", ""), fields.get("action", ""), fields.get("result", "")),
    )
    case_id = conn.execute("SELECT id FROM cases WHERE source = ?", (source,)).fetchone()["id"]
    conn.execute("DELETE FROM chunks WHERE case_id = ?", (case_id,))
    conn.executemany(
        "INSERT INTO chunks (case_id, ord, text, embedding) VALUES (?, ?, ?, ?)",
        [(case_id, i, t, np.asarray(v, dtype=np.float32).tobytes()) for i, (t, v) in enumerate(chunks)],
    )
    conn.commit()
    return case_id


def delete_sources(conn, sources):
    if not sources:
        return
    rows = conn.execute(
        f"SELECT id FROM cases WHERE source IN ({','.join('?' * len(sources))})", tuple(sources)
    ).fetchall()
    ids = [r["id"] for r in rows]
    conn.executemany("DELETE FROM chunks WHERE case_id = ?", [(i,) for i in ids])
    conn.executemany("DELETE FROM cases WHERE id = ?", [(i,) for i in ids])
    conn.commit()


def all_sources(conn):
    return {r["source"] for r in conn.execute("SELECT source FROM cases")}


# ──────────────────────────────────────────────────────────────
# インメモリ・インデックス（チャンク行列＋BM25）
# ──────────────────────────────────────────────────────────────
_LOCK = threading.Lock()
_INDEX = None


def _db_mtime():
    return os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0.0


def _build_index(conn):
    cases = conn.execute(
        "SELECT id, title, source, text, excerpt, industry, problem, action, result "
        "FROM cases ORDER BY id"
    ).fetchall()
    meta, id2pos = [], {}
    for pos, r in enumerate(cases):
        id2pos[r["id"]] = pos
        meta.append({k: r[k] for k in
                     ("id", "title", "source", "text", "excerpt", "industry", "problem", "action", "result")})

    rows = conn.execute("SELECT case_id, text, embedding FROM chunks ORDER BY case_id, ord").fetchall()
    chunk_text, chunk_case, embs = [], [], []
    for r in rows:
        pos = id2pos.get(r["case_id"])
        if pos is None:
            continue
        chunk_text.append(r["text"])
        chunk_case.append(pos)
        embs.append(np.frombuffer(r["embedding"], dtype=np.float32))

    C = len(meta)
    chunk_emb = np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32)
    chunk_by_case = [[] for _ in range(C)]
    for ci, pos in enumerate(chunk_case):
        chunk_by_case[pos].append(ci)

    d = chunk_emb.shape[1] if chunk_emb.size else 0
    case_vec = np.zeros((C, d), dtype=np.float32)
    for pos, cis in enumerate(chunk_by_case):
        if cis:
            v = chunk_emb[cis].mean(axis=0)
            n = np.linalg.norm(v)
            case_vec[pos] = v / n if n else v

    bm25 = None
    if HYBRID != "off" and chunk_text:
        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi([tokenize(t) for t in chunk_text])
        except Exception:  # noqa: BLE001
            bm25 = None

    return {
        "meta": meta, "case_vec": case_vec, "chunk_emb": chunk_emb,
        "chunk_text": chunk_text, "chunk_by_case": chunk_by_case, "bm25": bm25,
    }


def get_index():
    global _INDEX
    with _LOCK:
        mt = _db_mtime()
        if _INDEX is None or _INDEX["mtime"] != mt:
            conn = connect()
            init_db(conn)
            ix = _build_index(conn)
            conn.close()
            ix["mtime"] = _db_mtime()
            _INDEX = ix
        return _INDEX


def invalidate_cache():
    global _INDEX
    with _LOCK:
        _INDEX = None


# ──────────────────────────────────────────────────────────────
# ランキング（密 → ハイブリッド → リランク）
# ──────────────────────────────────────────────────────────────
def _ranks(order):
    """argsort 降順の並びから各要素の順位(0始まり)を返す。"""
    r = np.empty(len(order), dtype=np.int64)
    r[order] = np.arange(len(order))
    return r


def _rank_cases(ix, query):
    """事例を関連順に並べる。各要素 = {pos, cos, best}（cos=最良チャンクの類似度）。"""
    chunk_emb = ix["chunk_emb"]
    chunk_by_case = ix["chunk_by_case"]
    C = len(ix["meta"])
    if not C or chunk_emb.size == 0:
        return []

    # 任意: Azure クエリ拡張(HyDE)。密検索のみに反映し、BM25 は原クエリのまま。
    dense_query = query
    if azure_ai.expand_enabled():
        extra = azure_ai.expand_query(query)
        if extra:
            dense_query = query + "\n" + extra

    qv = embed([dense_query], "query")[0]
    if qv.shape[0] != chunk_emb.shape[1]:
        raise RuntimeError(
            "埋め込み次元が保存データと一致しません"
            f"（クエリ {qv.shape[0]} 次元 / 保存 {chunk_emb.shape[1]} 次元）。"
            "EMBED_BACKEND を変更した場合は `python ingest.py` で再取り込みしてください。"
        )
    chunk_cos = chunk_emb @ qv  # (M,)

    case_cos = np.full(C, -1.0)
    best = np.zeros(C, dtype=np.int64)
    for pos, cis in enumerate(chunk_by_case):
        if not cis:
            continue
        sub = chunk_cos[cis]
        j = int(np.argmax(sub))
        case_cos[pos] = float(sub[j])
        best[pos] = cis[j]

    # ハイブリッド（BM25 を RRF 融合）。融合順は「どれをリランクするか」の選別に使う。
    rank_dense = _ranks(np.argsort(-case_cos))
    case_bm = None
    if ix["bm25"] is not None:
        cb = np.asarray(ix["bm25"].get_scores(tokenize(query)))
        case_bm = np.zeros(C)
        for pos, cis in enumerate(chunk_by_case):
            if cis:
                case_bm[pos] = float(cb[cis].max())
        rank_bm = _ranks(np.argsort(-case_bm))
        fused = 1.0 / (RRF_K + rank_dense) + 1.0 / (RRF_K + rank_bm)
    else:
        fused = 1.0 / (RRF_K + rank_dense)

    cand = [int(p) for p in np.argsort(-fused)]

    # クロスエンコーダ・リランク（融合上位のみにスコアを付与）
    rr_by_pos = {}
    if rerank.available() and cand:
        topN = cand[:RERANK_TOP]
        texts = [ix["chunk_text"][int(best[pos])] for pos in topN]
        scores = rerank.rerank(query, texts)
        if scores is not None:
            for pos, sc in zip(topN, scores):
                rr_by_pos[pos] = float(sc)

    return [
        {"pos": pos, "cos": float(case_cos[pos]), "best": int(best[pos]),
         "rr": rr_by_pos.get(pos), "bm": (float(case_bm[pos]) if case_bm is not None else None)}
        for pos in cand
    ]


def _scored(ix, query: str, industry: str = ""):
    """候補に統一関連度 rel を付与し、rel 降順で返す（業種フィルタ済み）。"""
    out = []
    for c in _rank_cases(ix, query):
        if industry and ix["meta"][c["pos"]]["industry"] != industry:
            continue
        c["rel"] = candidate_relevance(c)
        out.append(c)
    out.sort(key=lambda c: -c["rel"])
    return out


def ranked_sources(query: str, industry: str = "", limit: int = 50):
    """評価用：閾値・top_kを無視した、関連度順の事例ソース列。"""
    ix = get_index()
    out = []
    for c in _scored(ix, query, industry):
        out.append(ix["meta"][c["pos"]]["source"])
        if len(out) >= limit:
            break
    return out


# ──────────────────────────────────────────────────────────────
# 検索（表示用）
# ──────────────────────────────────────────────────────────────
def search(query: str, top_k: int = 6, industry: str = "", min_rel: float | None = None):
    """足切り・表示%・並び順をすべて統一関連度 rel で判断する。"""
    if min_rel is None:
        min_rel = MIN_REL
    ix = get_index()
    if not ix["meta"] or not query.strip():
        return {"nodes": [], "edges": [], "hidden": 0}

    nodes, hidden = [], 0
    for c in _scored(ix, query, industry):
        rel = c["rel"]
        if rel < WEAK_REL:
            continue
        if rel < min_rel:
            hidden += 1
            continue
        if len(nodes) >= top_k:
            continue
        m = ix["meta"][c["pos"]]
        evidence = ix["chunk_text"][c["best"]]
        nodes.append({
            "id": m["id"], "title": m["title"], "source": m["source"], "excerpt": m["excerpt"],
            "industry": m["industry"], "problem": m["problem"], "action": m["action"], "result": m["result"],
            "score": c["cos"], "relevance": rel,
            "evidence": evidence[:300] + ("…" if len(evidence) > 300 else ""),
            "matched": matched_spans(query, evidence),
            "_idx": c["pos"],
        })

    edges = _edges(nodes, ix["case_vec"])
    for n in nodes:
        n.pop("_idx", None)
    return {"nodes": nodes, "edges": edges, "hidden": hidden}


def _edges(nodes, case_vec):
    pairs = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            sim = float(case_vec[nodes[a]["_idx"]] @ case_vec[nodes[b]["_idx"]])
            pairs.append((nodes[a]["id"], nodes[b]["id"], sim))
    if not pairs:
        return []
    sims = np.array([p[2] for p in pairs])
    thr = max(EDGE_FLOOR, float(np.percentile(sims, 70)))
    return [{"a": a, "b": b, "sim": s} for (a, b, s) in pairs if s >= thr]


def list_industries():
    return sorted({m["industry"] for m in get_index()["meta"] if m["industry"]})


def stats():
    ix = get_index()
    az = azure_ai.status()
    return {
        "count": len(ix["meta"]),
        "model": ("azure:" + os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "")) if az["embed"] else MODEL_NAME,
        "industries": sorted({m["industry"] for m in ix["meta"] if m["industry"]}),
        "hybrid": ix["bm25"] is not None,
        "reranker": rerank.available(),
        "azure": az,
    }
