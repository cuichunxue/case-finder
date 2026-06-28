"""テスト共通設定。

埋め込みモデルを軽量なスタブに差し替え、DBとdata/を一時ディレクトリに隔離する。
これにより、重いモデルのDLや本物のOCRなしで全体の配線を検証できる。
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# スタブ埋め込み用の語彙（出現回数を正規化してベクトル化）
VOCAB = [
    "離職", "定着", "若手", "メンター", "属人", "ナレッジ", "サポート", "検索",
    "離脱", "UI", "購入", "効率", "品質", "新人", "現場", "課題", "施策", "成果",
    "製造", "小売",
]


def _fake_embed(texts, kind):
    out = []
    for t in texts:
        v = np.array([t.count(w) for w in VOCAB], dtype=np.float32)
        n = np.linalg.norm(v)
        out.append(v / n if n else v)
    return np.vstack(out).astype(np.float32)


@pytest.fixture
def env(tmp_path, monkeypatch):
    import extract
    import ingest
    import search

    datadir = tmp_path / "data"
    datadir.mkdir()
    monkeypatch.setattr(search, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(search, "DATA_DIR", str(datadir))
    monkeypatch.setattr(ingest, "DATA_DIR", str(datadir))

    monkeypatch.setattr(search, "embed", _fake_embed)
    monkeypatch.setattr(extract, "embed", _fake_embed)
    monkeypatch.setattr(ingest, "embed", _fake_embed)

    extract._prototypes.cache_clear()
    extract._industry_protos.cache_clear()
    search.invalidate_cache()

    yield types.SimpleNamespace(
        search=search, extract=extract, ingest=ingest, datadir=datadir, tmp=tmp_path
    )

    search.invalidate_cache()
