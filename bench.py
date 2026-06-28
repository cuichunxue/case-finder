"""検索精度の評価ハーネス（Recall@k / MRR / nDCG）。

「LLM同等以上」を名乗るには測定が要る。eval.json（calibrate.py と同形式）で
ランキング品質を数値化し、ハイブリッド／リランカーの ON/OFF を A/B 比較できる。

使い方:
    python ingest.py
    cp eval.sample.json eval.json      # 無ければ
    python bench.py                    # 現設定で評価
    CASE_FINDER_HYBRID=off python bench.py     # 密のみ
    CASE_FINDER_RERANK=off python bench.py     # リランカー無効
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


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hits(query, rel_keys, k=10):
    """関連順のソース列に対し、各位置が正解かの真偽列（先頭k件）。"""
    srcs = search.ranked_sources(query, limit=k)
    out = []
    for s in srcs:
        hay = s.lower()
        out.append(any(key.lower() in hay for key in rel_keys))
    return out


def dcg(rels):
    return sum((1.0 if r else 0.0) / math.log2(i + 2) for i, r in enumerate(rels))


def main(argv):
    path = argv[0] if argv else os.path.join(search.BASE_DIR, "eval.json")
    if not os.path.exists(path):
        raise SystemExit(f"評価データがありません: {path}（eval.sample.json を参照）")
    data = load_eval(path)
    if not search.get_index()["meta"]:
        raise SystemExit("事例が未登録です。先に `python ingest.py` を実行してください。")

    ks = (1, 3, 5)
    agg = {f"recall@{k}": 0.0 for k in ks}
    agg["mrr"] = 0.0
    agg["ndcg@5"] = 0.0
    n = 0

    print(f"設定: hybrid={search.stats()['hybrid']}  reranker={search.stats()['reranker']}\n")
    for c in data:
        q, rel = c.get("query", ""), c.get("relevant", [])
        if not q or not rel:
            continue
        n += 1
        h = hits(q, rel, k=max(ks) if max(ks) >= 5 else 5)
        h5 = (h + [False] * 5)[:5]
        for k in ks:
            agg[f"recall@{k}"] += 1.0 if any(h[:k]) else 0.0
        rr = 0.0
        for i, ok in enumerate(h):
            if ok:
                rr = 1.0 / (i + 1)
                break
        agg["mrr"] += rr
        ideal = dcg([True] * min(len(rel), 5))
        agg["ndcg@5"] += (dcg(h5) / ideal) if ideal else 0.0
        mark = "✓" if (h and h[0]) else (" " if any(h) else "✗")
        print(f"  [{mark}] {q[:32]:<32}  上位: {''.join('●' if x else '·' for x in h5)}")

    if not n:
        raise SystemExit("有効な評価項目がありません。")

    print("\n=== スコア（{}クエリ平均）===".format(n))
    for k in ks:
        print(f"  Recall@{k}: {agg[f'recall@{k}'] / n:.3f}")
    print(f"  MRR     : {agg['mrr'] / n:.3f}")
    print(f"  nDCG@5  : {agg['ndcg@5'] / n:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
