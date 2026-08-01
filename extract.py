"""事例テキストを「課題 / 施策 / 成果」に分ける（BERT系エンコーダで分類）。

生成AI（LLM）は使いません。検索と同じ埋め込みモデル（multilingual-e5＝
XLM-RoBERTa／BERT系エンコーダ）で各文をベクトル化し、ラベルの代表文（プロトタイプ）
との意味的な近さ（コサイン類似度）で分類する「ゼロショット分類」です。

優先順位:
  1) 行頭に「課題:」「施策:」「成果:」等の見出しがあればそれを採用（最も確実）
  2) 見出しが無い行は BERT 埋め込みで最も近いラベルに割り当て
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from search import embed

# (DB列名, 表示名, 見出しとして認識する語, 分類用プロトタイプ文)
LABELS = [
    (
        "problem",
        "課題",
        ["課題", "問題", "問題点", "背景", "悩み", "as-is", "現状"],
        ["これは解決したい課題・問題点です", "困っていること、うまくいかない点、ボトルネック"],
    ),
    (
        "action",
        "施策",
        ["施策", "対策", "対応", "取り組み", "実施", "導入", "解決策", "アプローチ", "to-be"],
        ["これは実施した施策・対策です", "導入した仕組みや取り組み、改善のために行った内容"],
    ),
    (
        "result",
        "成果",
        ["成果", "結果", "効果", "実績", "アウトカム", "kpi"],
        ["これは得られた成果・効果です", "改善した数値、削減・向上・短縮といった結果"],
    ),
]

_HEAD_RE = {
    col: re.compile(
        r"^\s*[\-・●○◆■]?\s*(" + "|".join(map(re.escape, heads)) + r")\s*[:：\)）.。]\s*",
        re.IGNORECASE,
    )
    for col, _, heads, _ in LABELS
}

# e5 は正規化済みベクトル同士でも全体的に高めに出るため、しきい値は高め
CLASSIFY_THRESHOLD = 0.80

# 業種の候補（環境変数 CASE_FINDER_INDUSTRIES でカンマ区切り上書き可）
import os as _os

INDUSTRIES = [
    s.strip()
    for s in _os.environ.get(
        "CASE_FINDER_INDUSTRIES",
        "IT・SaaS,製造,小売・EC,サービス,金融,医療・ヘルスケア,建設・不動産,教育,物流,公共・自治体",
    ).split(",")
    if s.strip()
]
# 業種推定の最低類似度（これ未満は「業種なし」にして誤分類を避ける）
INDUSTRY_THRESHOLD = float(_os.environ.get("CASE_FINDER_INDUSTRY_THRESHOLD", "0.80"))


@lru_cache(maxsize=1)
def _prototypes():
    """各ラベルのプロトタイプ平均ベクトル（正規化済み）。"""
    cols, mats = [], []
    for col, _, _, protos in LABELS:
        v = embed(protos, "passage").mean(axis=0)
        n = np.linalg.norm(v)
        mats.append(v / n if n else v)
        cols.append(col)
    return cols, np.vstack(mats).astype(np.float32)


def _split_lines(text: str):
    lines = []
    for raw in text.replace("。", "。\n").splitlines():
        s = raw.strip()
        if len(s) >= 4:  # ノイズな短い行は除外
            lines.append(s)
    return lines


def _match_heading(line: str):
    for col, _, _, _ in LABELS:
        m = _HEAD_RE[col].match(line)
        if m:
            return col, line[m.end():].strip()
    return None, line


def extract_fields(text: str) -> dict:
    """text を {'problem','action','result'} に振り分けて返す。"""
    out = {"problem": [], "action": [], "result": []}
    lines = _split_lines(text)
    if not lines:
        return {k: "" for k in out}

    # 1) 見出しのある行を先に確定
    pending = []
    for line in lines:
        col, body = _match_heading(line)
        if col and body:
            out[col].append(body)
        elif col and not body:
            pending.append(("__heading__:" + col, line))
        else:
            pending.append((None, line))

    # 2) 残りを BERT 埋め込みで分類
    free = [ln for tag, ln in pending if tag is None]
    if free:
        cols, proto = _prototypes()
        vecs = embed(free, "passage")  # 正規化済み
        sims = vecs @ proto.T  # (M, 3)
        best = sims.argmax(axis=1)
        for i, ln in enumerate(free):
            if sims[i, best[i]] >= CLASSIFY_THRESHOLD:
                out[cols[best[i]]].append(ln)

    return {k: "\n".join(v).strip() for k, v in out.items()}


@lru_cache(maxsize=1)
def _industry_protos():
    # 業種名をそのままプロトタイプ文として埋め込む
    return embed([f"これは{name}業界の事例です" for name in INDUSTRIES], "passage")


def classify_industry(text: str) -> str:
    """事例本文から業種をBERT埋め込みで推定する（生成AI不使用）。

    最も近い業種を返す。どれにも十分近くなければ空文字（業種なし）。
    """
    if not INDUSTRIES or not text.strip():
        return ""
    head = " ".join(text.split())[:600]  # 冒頭中心に判定
    v = embed([head], "passage")[0]
    sims = _industry_protos() @ v
    best = int(sims.argmax())
    return INDUSTRIES[best] if sims[best] >= INDUSTRY_THRESHOLD else ""
