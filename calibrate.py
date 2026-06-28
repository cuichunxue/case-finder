"""しきい値キャリブレーション・ツール。

実コーパスに対して MIN_SCORE / REL_FLOOR / REL_CEIL の妥当な値を求めます。
当て推量の閾値を、あなたの事例に合わせて校正するためのものです。

準備:
  1) 事例を取り込む:  python ingest.py
  2) 評価データを作る: eval.json（eval.sample.json を参考に）
       [{"query": "若手がすぐ辞める", "relevant": ["若手の早期離職"]}, ...]
       relevant は「正解事例の source か title に含まれる部分文字列」のリスト
  3) 実行:            python calibrate.py [eval.json]

出力:
  - 閾値ごとの Precision / Recall / F1（最良F1の閾値を推奨）
  - 正解／不正解スコアの分布（REL_FLOOR/CEIL の目安）
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import search


def load_eval(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise SystemExit("評価データが空、または配列ではありません。")
    return data


def is_relevant(meta_row, keys) -> bool:
    hay = (meta_row["source"] + " " + meta_row["title"]).lower()
    return any(k.lower() in hay for k in keys)


def main(argv):
    eval_path = argv[0] if argv else os.path.join(search.BASE_DIR, "eval.json")
    if not os.path.exists(eval_path):
        raise SystemExit(
            f"評価データが見つかりません: {eval_path}\n"
            f"eval.sample.json をコピーして eval.json を作成してください。"
        )
    cases = load_eval(eval_path)

    meta, matrix = search.get_index()
    if not meta:
        raise SystemExit("事例が未登録です。先に `python ingest.py` を実行してください。")

    # 全クエリ×全事例のスコアを集め、正解／不正解に仕分け
    pos, neg = [], []  # 正解事例のスコア, 不正解事例のスコア
    per_query = []
    for c in cases:
        q, rel = c.get("query", ""), c.get("relevant", [])
        if not q or not rel:
            continue
        qv = search.embed([q], "query")[0]
        scores = matrix @ qv
        labels = np.array([is_relevant(m, rel) for m in meta])
        if not labels.any():
            print(f"  ⚠ クエリ『{q}』の正解事例がDBに見つかりません（relevant指定を確認）")
        pos.extend(scores[labels].tolist())
        neg.extend(scores[~labels].tolist())
        per_query.append((q, scores, labels))

    if not pos:
        raise SystemExit("正解スコアが集まりませんでした。relevant の指定を見直してください。")

    pos_a, neg_a = np.array(pos), np.array(neg)

    print("\n=== スコア分布 ===")
    print(f"  正解  : 件数{len(pos_a):4d}  中央値 {np.median(pos_a):.3f}  "
          f"5%点 {np.percentile(pos_a,5):.3f}  95%点 {np.percentile(pos_a,95):.3f}")
    print(f"  不正解: 件数{len(neg_a):4d}  中央値 {np.median(neg_a):.3f}  "
          f"5%点 {np.percentile(neg_a,5):.3f}  95%点 {np.percentile(neg_a,95):.3f}")

    print("\n=== 閾値スイープ（Precision / Recall / F1）===")
    best = (0.0, -1.0)  # (threshold, f1)
    for thr in np.round(np.arange(0.70, 0.951, 0.01), 2):
        tp = fp = fn = 0
        for _, scores, labels in per_query:
            pred = scores >= thr
            tp += int(np.sum(pred & labels))
            fp += int(np.sum(pred & ~labels))
            fn += int(np.sum(~pred & labels))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        mark = ""
        if f1 > best[1]:
            best = (float(thr), f1)
        print(f"  thr {thr:.2f}  P {prec:.2f}  R {rec:.2f}  F1 {f1:.2f}")

    # 推奨値の提示
    rel_floor = round(float(np.percentile(neg_a, 90)), 2)   # 不正解の上側 ≒ 0% 表示の下限
    rel_ceil = round(float(np.percentile(pos_a, 75)), 2)    # 正解の中位 ≒ 100% 表示の上限
    print("\n=== 推奨値 ===")
    print(f"  CASE_FINDER_MIN_SCORE = {best[0]:.2f}   （最良F1 {best[1]:.2f}）")
    print(f"  CASE_FINDER_REL_FLOOR = {rel_floor:.2f}")
    print(f"  CASE_FINDER_REL_CEIL  = {max(rel_ceil, rel_floor + 0.05):.2f}")
    print("\n環境変数に設定して app.py / ingest.py を再実行すると反映されます。")


if __name__ == "__main__":
    main(sys.argv[1:])
