"""実行デバイスの解決（CPU / GPU）。

CASE_FINDER_DEVICE = auto（既定）| cpu | cuda | mps
auto は GPU が使えれば自動採用。torch 未導入なら cpu。
埋め込み・リランカー・OCR で共通利用する。
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def resolve() -> str:
    d = os.environ.get("CASE_FINDER_DEVICE", "auto").lower()
    if d in ("cpu", "cuda", "mps"):
        return d
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def use_gpu() -> bool:
    return resolve() != "cpu"
