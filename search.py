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
import cache
import device
import metrics
import rerank

# SQLite のロック待ち時間（ミリ秒）。取り込みと書き込みの競合を吸収する。
DB_TIMEOUT_MS = int(os.environ.get("CASE_FINDER_DB_TIMEOUT", "5000"))

# 埋め込みバックエンド: local（既定・完全ローカル）/ azure（Azure OpenAI 埋め込み）
EMBED_BACKEND = os.environ.get("CASE_FINDER_EMBED_BACKEND", "local").lower()

# 近似最近傍(ANN)。大規模時のみ自動的に密検索を高速化（hnswlib）。
ANN = os.environ.get("CASE_FINDER_ANN", "auto").lower()  # auto|on|off
ANN_MIN = int(os.environ.get("CASE_FINDER_ANN_MIN", "2000"))  # この件数以上で有効化
ANN_K = int(os.environ.get("CASE_FINDER_ANN_K", "200"))       # 取得する近傍チャンク数

# 結果キャッシュ（同一クエリの再計算・Azure課金を削減）
CACHE_SIZE = int(os.environ.get("CASE_FINDER_CACHE_SIZE", "256"))
CACHE_TTL = float(os.environ.get("CASE_FINDER_CACHE_TTL", "300"))
_RESULT_CACHE = cache.TTLCache(CACHE_SIZE, CACHE_TTL)

# ──────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB と事例フォルダは環境変数で差し替え可能（保存先を変えたいとき用）
DB_PATH = os.environ.get("CASE_FINDER_DB", os.path.join(BASE_DIR, "cases.db"))
DATA_DIR = os.environ.get("CASE_FINDER_DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

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

# 近重複の抑制（検索結果から事実上同一の事例を1件に集約）
DEDUP = os.environ.get("CASE_FINDER_DEDUP", "on").lower() != "off"
DEDUP_THRESHOLD = float(os.environ.get("CASE_FINDER_DEDUP_THRESHOLD", "0.98"))


# ──────────────────────────────────────────────────────────────
# 埋め込みモデル（遅延ロード）
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device=device.resolve())


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
# トークナイザ: char（既定・依存なし）/ sudachi（要 sudachipy）/ auto
TOKENIZER = os.environ.get("CASE_FINDER_TOKENIZER", "char").lower()
_sudachi = {"obj": None, "failed": False}


def _sudachi_tokens(text: str):
    if _sudachi["failed"]:
        return None
    try:
        if _sudachi["obj"] is None:
            from sudachipy import dictionary, tokenizer as _tk

            _sudachi["obj"] = (dictionary.Dictionary().create(), _tk.Tokenizer.SplitMode.C)
        tok, mode = _sudachi["obj"]
        return [m.normalized_form() for m in tok.tokenize(text, mode)
                if m.surface().strip()]
    except Exception:  # noqa: BLE001
        _sudachi["failed"] = True
        return None


def _char_tokens(text: str):
    toks = re.findall(r"[a-z0-9][a-z0-9\-\.]*", text)
    for run in re.findall(f"[{_CJK}]+", text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks += [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


def tokenize(text: str):
    """語彙一致用トークン。既定は文字bigram（依存なし）。

    形態素解析(Sudachi)を使うと語境界が正確になり、BM25のIDFが安定する。
    CASE_FINDER_TOKENIZER=sudachi|auto かつ sudachipy 導入時に有効。
    """
    text = text.lower()
    if TOKENIZER in ("sudachi", "auto"):
        m = _sudachi_tokens(text)
        if m is not None:
            return m
        if TOKENIZER == "sudachi":  # 明示指定で未導入なら char にフォールバック
            pass
    return _char_tokens(text)


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
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT_MS / 1000.0)
    conn.row_factory = sqlite3.Row
    # 同時書き込みでの "database is locked" を抑える設定
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={DB_TIMEOUT_MS}")
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      REAL NOT NULL,
            query   TEXT NOT NULL,
            case_id INTEGER,
            vote    TEXT NOT NULL
        )
        """
    )
    conn.commit()


def record_feedback(query: str, case_id, vote: str):
    """検索結果への👍/👎を記録（将来の調整用。索引には影響しない）。"""
    import time as _t

    conn = connect()
    init_db(conn)
    conn.execute(
        "INSERT INTO feedback (ts, query, case_id, vote) VALUES (?, ?, ?, ?)",
        (_t.time(), query, case_id, "up" if vote == "up" else "down"),
    )
    conn.commit()
    conn.close()
    metrics.incr("feedback_" + ("up" if vote == "up" else "down"))


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

    ann = _build_ann(chunk_emb)

    return {
        "meta": meta, "case_vec": case_vec, "chunk_emb": chunk_emb,
        "chunk_text": chunk_text, "chunk_by_case": chunk_by_case, "bm25": bm25, "ann": ann,
    }


def _build_ann(chunk_emb):
    """大規模時のみ hnswlib で密検索を近似高速化（小規模・未導入はNone=総当たり）。"""
    if ANN == "off" or chunk_emb.size == 0 or chunk_emb.shape[0] < ANN_MIN:
        return None
    try:
        import hnswlib

        n, d = chunk_emb.shape
        idx = hnswlib.Index(space="cosine", dim=d)
        idx.init_index(max_elements=n, ef_construction=200, M=16)
        idx.add_items(chunk_emb, np.arange(n))
        idx.set_ef(max(ANN_K, 64))
        return idx
    except Exception:  # noqa: BLE001
        return None


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
    _RESULT_CACHE.clear()


def index_version():
    """キャッシュキー用のインデックス版（DB更新で変わる）。"""
    return _db_mtime()


# ──────────────────────────────────────────────────────────────
# ランキング（密 → ハイブリッド → リランク）
# ──────────────────────────────────────────────────────────────
def _ranks(order):
    """argsort 降順の並びから各要素の順位(0始まり)を返す。"""
    r = np.empty(len(order), dtype=np.int64)
    r[order] = np.arange(len(order))
    return r


def _rank_cases(ix, query, use_ann: bool = True):
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

    M = chunk_emb.shape[0]
    if use_ann and ix.get("ann") is not None:
        # ANN: 上位近傍チャンクだけ取得（取得外は -1 とし、対象事例を絞る）
        k = min(ANN_K, M)
        labels, dists = ix["ann"].knn_query(qv, k=k)
        chunk_cos = np.full(M, -1.0, dtype=np.float32)
        chunk_cos[labels[0]] = (1.0 - dists[0]).astype(np.float32)
        metrics.incr("ann_queries")
    else:
        chunk_cos = chunk_emb @ qv  # (M,) 総当たり（業種フィルタ時はこちらで正確に）

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
    # 業種フィルタ時は ANN を使わず総当たりにして、ニッチ業種の取りこぼしを防ぐ
    use_ann = not industry
    out = []
    for c in _rank_cases(ix, query, use_ann=use_ann):
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
    if not query.strip():
        return {"nodes": [], "edges": [], "hidden": 0, "duplicates": 0}

    ckey = (query, industry, top_k, round(min_rel, 6), _db_mtime())
    cached = _RESULT_CACHE.get(ckey)
    if cached is not None:
        return cached

    metrics.incr("searches")
    _timer = metrics.timed("search")
    _timer.__enter__()
    ix = get_index()
    if not ix["meta"]:
        _timer.__exit__()
        return {"nodes": [], "edges": [], "hidden": 0, "duplicates": 0}

    case_vec = ix["case_vec"]
    nodes, hidden, duplicates = [], 0, 0
    for c in _scored(ix, query, industry):
        rel = c["rel"]
        if rel < WEAK_REL:
            continue
        if rel < min_rel:
            hidden += 1
            continue
        if len(nodes) >= top_k:
            continue
        # 近重複（事実上同一の事例）は最上位の1件だけ残す
        if DEDUP and any(
            float(case_vec[c["pos"]] @ case_vec[n["_idx"]]) >= DEDUP_THRESHOLD for n in nodes
        ):
            duplicates += 1
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
    result = {"nodes": nodes, "edges": edges, "hidden": hidden, "duplicates": duplicates}
    _RESULT_CACHE.put(ckey, result)
    _timer.__exit__()
    return result


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


def find_duplicates(thr: float | None = None):
    """近重複の事例群を列挙する（case_vec のコサインが閾値以上）。"""
    thr = DEDUP_THRESHOLD if thr is None else thr
    ix = get_index()
    cv, meta = ix["case_vec"], ix["meta"]
    C = len(meta)
    if C < 2 or cv.size == 0:
        return []
    sims = cv @ cv.T
    seen, groups = set(), []
    for i in range(C):
        if i in seen:
            continue
        grp = [i]
        for j in range(i + 1, C):
            if j not in seen and float(sims[i, j]) >= thr:
                grp.append(j)
                seen.add(j)
        if len(grp) > 1:
            seen.update(grp)
            groups.append({"ids": [meta[g]["id"] for g in grp],
                           "titles": [meta[g]["title"] for g in grp]})
    return groups


def ann_recall(k: int = 10, n_probe: int = 200, seed: int = 0):
    """ANN近傍と総当たりの overlap@k を実測（ANN未使用なら None）。"""
    ix = get_index()
    if ix.get("ann") is None:
        return None
    emb = ix["chunk_emb"]
    M = emb.shape[0]
    if M == 0:
        return None
    rng = np.random.default_rng(seed)
    probes = rng.choice(M, size=min(n_probe, M), replace=False)
    kk = min(k, M)
    rec = []
    for p in probes:
        q = emb[p]
        brute = set(np.argsort(-(emb @ q))[:kk].tolist())
        labels, _ = ix["ann"].knn_query(q, k=kk)
        rec.append(len(brute & set(int(x) for x in labels[0])) / kk)
    return float(np.mean(rec)) if rec else None


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
        "ann": ix.get("ann") is not None,
        "device": device.resolve(),
        "azure": az,
    }
