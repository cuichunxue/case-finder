"""検索精度の評価ハーネス（段階的関連度 / Recall@k / MRR / nDCG / 信頼区間）。

「LLM同等以上」を名乗るには測定が要る。eval.json で品質を数値化し、
ハイブリッド／リランカーの ON/OFF を A/B 比較できる。指標計算は純関数として
切り出してありテスト可能。

eval.json の形式（2通り対応）:
  - 二値:   {"query": "...", "relevant": ["タイトル断片", ...]}
  - 段階的: {"query": "...", "relevant": {"断片A": 2, "断片B": 1}}   # 0-3 等の関連度

使い方:
    python ingest.py
    cp eval.sample.json eval.json
    python bench.py
    CASE_FINDER_HYBRID=off python bench.py     # 密のみと比較
    CASE_FINDER_RERANK=off python bench.py     # リランカー無しと比較
"""

from __future__ import annotations

import json
import math
import os
import sys

import azure_ai
import search

# 評価中は Azure クエリ拡張(HyDE)を無効化（静かな課金と非決定性を避ける）
azure_ai.EXPAND_ON = False


# ──────────────────────────────────────────────────────────────
# 純粋な指標関数（テスト可能）
# ──────────────────────────────────────────────────────────────
def grades_for(ranked_sources, rel_map):
    """関連順ソース列を、各位置の関連度（grade）の列に変換する。"""
    out = []
    for s in ranked_sources:
        hay = s.lower()
        g = 0
        for key, grade in rel_map.items():
            if key.lower() in hay:
                g = max(g, grade)
        out.append(g)
    return out


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg(grades, ideal_grades, k=5):
    idcg = dcg(sorted(ideal_grades, reverse=True)[:k])
    return (dcg(grades[:k]) / idcg) if idcg else 0.0


def recall_at_k(grades, n_relevant, k):
    if not n_relevant:
        return 0.0
    return sum(1 for g in grades[:k] if g > 0) / n_relevant


def mrr(grades):
    for i, g in enumerate(grades):
        if g > 0:
            return 1.0 / (i + 1)
    return 0.0


def query_metrics(grades, rel_grades):
    n_rel = sum(1 for g in rel_grades if g > 0)
    return {
        "recall@1": recall_at_k(grades, n_rel, 1),
        "recall@3": recall_at_k(grades, n_rel, 3),
        "recall@5": recall_at_k(grades, n_rel, 5),
        "mrr": mrr(grades),
        "ndcg@5": ndcg(grades, rel_grades, 5),
        "hit": 1.0 if (grades and grades[0] > 0) else 0.0,
    }


def bootstrap_ci(values, iters=1000, seed=0):
    """平均のブートストラップ95%信頼区間。"""
    import numpy as np

    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(iters, len(v)))].mean(axis=1)
    return (float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ──────────────────────────────────────────────────────────────
# 実行
# ──────────────────────────────────────────────────────────────
def _normalize_rel(rel):
    if isinstance(rel, dict):
        return {str(k): int(v) for k, v in rel.items()}
    return {str(k): 1 for k in rel}  # 二値→grade1


def main(argv):
    path = argv[0] if argv else os.path.join(search.BASE_DIR, "eval.json")
    if not os.path.exists(path):
        raise SystemExit(f"評価データがありません: {path}（eval.sample.json を参照）")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not search.get_index()["meta"]:
        raise SystemExit("事例が未登録です。先に `python ingest.py` を実行してください。")

    per = {m: [] for m in ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "hit")}
    print(f"設定: hybrid={search.stats()['hybrid']}  reranker={search.stats()['reranker']}"
          f"  device={search.stats()['device']}\n")
    n = 0
    for c in data:
        q, rel = c.get("query", ""), c.get("relevant")
        if not q or not rel:
            continue
        rel_map = _normalize_rel(rel)
        n += 1
        ranked = search.ranked_sources(q, limit=10)
        grades = grades_for(ranked, rel_map)
        rel_grades = list(rel_map.values())
        qm = query_metrics(grades, rel_grades)
        for k, v in qm.items():
            per[k].append(v)
        mark = "✓" if qm["hit"] else (" " if any(g > 0 for g in grades) else "✗")
        viz = "".join(("●" if g > 0 else "·") for g in grades[:5])
        print(f"  [{mark}] {q[:30]:<30} {viz}")

    if not n:
        raise SystemExit("有効な評価項目がありません。")

    print(f"\n=== スコア（{n}クエリ・平均 [95%CI]）===")
    for k in ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"):
        mean, lo, hi = bootstrap_ci(per[k])
        print(f"  {k:<9}: {mean:.3f}  [{lo:.3f}, {hi:.3f}]")
    if n < 20:
        print("\n注意: クエリ数が少なく信頼区間が広いです。実務では50〜100クエリを推奨。")


if __name__ == "__main__":
    main(sys.argv[1:])
