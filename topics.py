"""事例コーパスのトピック地図（テキストマインドマップ）。

検索とは別に、事例全体を俯瞰するための解析:
  - 事例埋め込み（case_vec）を k-means でクラスタリング
  - 各クラスタの特徴語を c-TF-IDF で抽出（読みやすい日本語n-gram＋英数語）
  - PCA で2次元に射影して地図化
  - kNN コサインで事例間エッジ（つながり）

外部依存なし（numpy のみ）。BERTopic 等が使える環境ではより高度化も可能だが、
ここでは追加依存ゼロで「どんな事例群があるか」を可視化する。
"""

from __future__ import annotations

import math
import re

import numpy as np

_CJK = r"぀-ヿ一-鿿ｦ-ﾟ"
_cache = {}


# ──────────────────────────────────────────────────────────────
# クラスタリング（k-means, コサイン）
# ──────────────────────────────────────────────────────────────
def _kmeans(X, k, seed=0, iters=60):
    n = X.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    # k-means++ 初期化（コサイン距離）
    idx = [int(rng.integers(n))]
    for _ in range(1, k):
        sims = X @ X[idx].T
        d = np.clip(1.0 - sims.max(axis=1), 0, None)
        s = d.sum()
        idx.append(int(rng.choice(n, p=d / s)) if s > 0 else int(rng.integers(n)))
    C = X[idx].copy()
    labels = np.full(n, -1)
    for it in range(iters):
        new = (X @ C.T).argmax(axis=1)
        if it > 0 and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = X[labels == j]
            if len(m):
                c = m.mean(axis=0)
                nrm = np.linalg.norm(c)
                C[j] = c / nrm if nrm else C[j]
    return labels


def _auto_k(n):
    if n <= 3:
        return max(1, n)
    return int(max(2, min(12, round(math.sqrt(n / 2)))))


# ──────────────────────────────────────────────────────────────
# PCA 2D
# ──────────────────────────────────────────────────────────────
def _pca2(X):
    if X.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    Xc = X - X.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comp = Xc @ Vt[:2].T
    except Exception:  # noqa: BLE001
        comp = Xc[:, :2]
    if comp.shape[1] < 2:
        comp = np.hstack([comp, np.zeros((comp.shape[0], 2 - comp.shape[1]))])
    mn = comp.min(axis=0)
    rng = comp.max(axis=0) - mn
    rng[rng == 0] = 1.0
    return (comp - mn) / rng  # [0,1] に正規化


# ──────────────────────────────────────────────────────────────
# c-TF-IDF キーワード
# ──────────────────────────────────────────────────────────────
def _ngrams(text, cap=800):
    text = text[:cap].lower()
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9\-\.]+", text):
        if len(w) >= 2:
            out.append(w)
    for run in re.findall(f"[{_CJK}]{{2,}}", text):
        L = len(run)
        for n in range(2, min(6, L) + 1):
            for i in range(L - n + 1):
                out.append(run[i:i + n])
    return out


def _cluster_keywords(texts_per_cluster, topn=6):
    from collections import Counter

    counts = [Counter(_ngrams(t)) for t in texts_per_cluster]
    # クラスタ内で1回しか出ない n-gram は刈る（ノイズ低減）
    counts = [Counter({t: c for t, c in cc.items() if c >= 2}) or cc for cc in counts]
    totals = [sum(cc.values()) or 1 for cc in counts]
    global_count = Counter()
    for cc in counts:
        global_count.update(cc)
    A = sum(totals) / max(1, len(counts))  # クラスタ当たり平均語数

    keywords = []
    for ci, cc in enumerate(counts):
        scored = []
        for t, c in cc.items():
            tf = c / totals[ci]
            idf = math.log(1 + A / global_count[t])
            scored.append((tf * idf, t))
        scored.sort(reverse=True)
        # 上位から、既選択語の部分文字列でないものを採用（最大長優先で読みやすく）
        picked = []
        for _, t in scored:
            if any(t != o and t in o for o in picked):
                continue
            picked = [o for o in picked if not (o != t and o in t)]
            picked.append(t)
            if len(picked) >= topn:
                break
        keywords.append(picked)
    return keywords


# ──────────────────────────────────────────────────────────────
# 地図の構築
# ──────────────────────────────────────────────────────────────
def _edges(case_vec, knn=3, thr=0.5):
    n = case_vec.shape[0]
    if n < 2:
        return []
    sims = case_vec @ case_vec.T
    np.fill_diagonal(sims, -1.0)
    seen, edges = set(), []
    for i in range(n):
        for j in np.argsort(-sims[i])[:knn]:
            j = int(j)
            s = float(sims[i, j])
            if s < thr:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"a": i, "b": j, "sim": s})
    return edges


def build_map(ix, k=None, seed=0):
    """ix=search.get_index() からトピック地図を構築して返す。"""
    meta = ix["meta"]
    cv = ix["case_vec"]
    n = len(meta)
    if n == 0 or cv.size == 0:
        return {"clusters": [], "cases": [], "edges": []}

    ck = (ix.get("mtime"), k, seed, n)
    if ck in _cache:
        return _cache[ck]

    k = k or _auto_k(n)
    labels = _kmeans(cv, k, seed=seed)
    coords = _pca2(cv)

    texts_per_cluster, members = [], []
    uniq = sorted(set(int(x) for x in labels))
    for cid in uniq:
        idxs = [i for i in range(n) if labels[i] == cid]
        members.append(idxs)
        blob = "\n".join(
            " ".join(filter(None, [meta[i]["title"], meta[i]["problem"],
                                   meta[i]["action"], meta[i]["result"], meta[i]["excerpt"]]))
            for i in idxs
        )
        texts_per_cluster.append(blob)
    kws = _cluster_keywords(texts_per_cluster)

    clusters = []
    for ord_i, cid in enumerate(uniq):
        clusters.append({
            "id": ord_i,
            "size": len(members[ord_i]),
            "keywords": kws[ord_i],
            "cases": [{"id": meta[i]["id"], "title": meta[i]["title"],
                       "industry": meta[i]["industry"]} for i in members[ord_i]],
        })
    remap = {cid: ord_i for ord_i, cid in enumerate(uniq)}
    cases = [{
        "id": meta[i]["id"], "title": meta[i]["title"], "industry": meta[i]["industry"],
        "cluster": remap[int(labels[i])],
        "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4),
    } for i in range(n)]

    result = {"clusters": clusters, "cases": cases, "edges": _edges(cv)}
    _cache.clear()
    _cache[ck] = result
    return result
