"""クロスエンコーダ・リランカー（生成AI不使用・BERT系）。

密検索で集めた上位候補を、(クエリ, 該当チャンク) のペアでスコアリングし直す。
検索精度の最大のレバーで、LLM をリランカーに使うより速く安く、精度は同等以上を狙える。

  CASE_FINDER_RERANK   = auto (既定) | off
  CASE_FINDER_RERANKER = モデル名（既定は軽量な日本語クロスエンコーダ）

モデルが未導入/DL不可なら自動的に無効化し、密検索のみで動作する（壊れない）。
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("CASE_FINDER_RERANK", "auto").lower()  # auto|off
MODEL_NAME = os.environ.get(
    "CASE_FINDER_RERANKER", "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1"
)

_state = {"model": None, "failed": False}


def _load():
    if ENABLED == "off" or _state["failed"]:
        return None
    if _state["model"] is not None:
        return _state["model"]
    try:
        import device
        from sentence_transformers import CrossEncoder

        _state["model"] = CrossEncoder(MODEL_NAME, max_length=512, device=device.resolve())
        return _state["model"]
    except Exception:  # noqa: BLE001
        _state["failed"] = True  # 一度失敗したら以後は試さない
        return None


def available() -> bool:
    return _load() is not None


def name() -> str:
    return MODEL_NAME if available() else "なし"


def rerank(query: str, texts: list[str]):
    """各 text の関連スコア（高いほど関連）。利用不可なら None。"""
    model = _load()
    if model is None or not texts:
        return None
    try:
        scores = model.predict([(query, t) for t in texts])
        return [float(s) for s in scores]
    except Exception:  # noqa: BLE001
        return None
