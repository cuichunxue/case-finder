"""スモークテスト（スタブ埋め込み）。

実モデル/OCRなしで、抽出・閾値/弱い候補・prune・API・認証の配線を検証する。
"""

from __future__ import annotations

import base64
import io
import math
import time

import numpy as np


def _unit(cos):
    """[1,0] との内積（コサイン）が cos になる単位ベクトル。"""
    return np.array([cos, math.sqrt(max(0.0, 1 - cos * cos))], dtype=np.float32)


# ── 課題/施策/成果の抽出（見出しパス）──
def test_extract_fields_headings(env):
    txt = "課題: 若手の離職が多い\n施策: メンター制度を導入\n成果: 離職率が改善"
    f = env.extract.extract_fields(txt)
    assert "離職が多い" in f["problem"]
    assert "メンター" in f["action"]
    assert "改善" in f["result"]


# ── 相対関連度の変換 ──
def test_relevance_mapping(env):
    s = env.search
    assert s.relevance(s.REL_CEIL) == 1.0
    assert s.relevance(s.REL_FLOOR) == 0.0
    assert s.relevance(s.REL_FLOOR - 0.1) == 0.0  # 下限未満は0でクランプ


# ── 閾値・弱い候補(hidden)・loose ──
def test_search_threshold_and_hidden(env, monkeypatch):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    s.upsert_case(conn, "A", "a.txt", "t", "e", "", {}, _unit(1.00))
    s.upsert_case(conn, "B", "b.txt", "t", "e", "", {}, _unit(0.85))
    s.upsert_case(conn, "C", "c.txt", "t", "e", "", {}, _unit(0.78))
    conn.close()
    s.invalidate_cache()

    monkeypatch.setattr(s, "embed", lambda texts, kind: np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(s, "MIN_SCORE", 0.80)
    monkeypatch.setattr(s, "WEAK_FLOOR", 0.75)

    r = s.search("q", top_k=6)
    assert [n["title"] for n in r["nodes"]] == ["A", "B"]
    assert r["hidden"] == 1  # C は閾値未満だが WEAK_FLOOR 以上

    # loose（弱い候補も表示）で C が出る
    r2 = s.search("q", top_k=6, min_score=s.WEAK_FLOOR)
    assert "C" in [n["title"] for n in r2["nodes"]]


# ── 業種フィルタ ──
def test_industry_filter(env, monkeypatch):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    s.upsert_case(conn, "M", "m.txt", "t", "e", "製造", {}, _unit(0.95))
    s.upsert_case(conn, "R", "r.txt", "t", "e", "小売・EC", {}, _unit(0.95))
    conn.close()
    s.invalidate_cache()
    monkeypatch.setattr(s, "embed", lambda texts, kind: np.array([[1.0, 0.0]], dtype=np.float32))
    r = s.search("q", top_k=6, industry="製造", min_score=0.0)
    assert [n["title"] for n in r["nodes"]] == ["M"]
    assert s.list_industries() == ["小売・EC", "製造"]


# ── prune（消えたファイルのレコード掃除）──
def test_prune(env):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    s.upsert_case(conn, "X", "data/gone.txt", "t", "e", "", {}, _unit(1.0))
    assert "data/gone.txt" in s.all_sources(conn)
    s.delete_sources(conn, {"data/gone.txt"})
    assert "data/gone.txt" not in s.all_sources(conn)
    conn.close()


# ── キャッシュ（明示無効化）──
def test_cache_invalidate(env):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    s.upsert_case(conn, "A", "a.txt", "t", "e", "", {}, _unit(1.0))
    conn.close()
    s.invalidate_cache()
    meta1, _ = s.get_index()
    assert len(meta1) == 1


# ── API: stats / search ──
def test_api_stats_and_search(env, monkeypatch):
    import app

    s = env.search
    conn = s.connect()
    s.init_db(conn)
    s.upsert_case(conn, "A", "a.txt", "t", "e", "製造", {}, _unit(1.0))
    conn.close()
    s.invalidate_cache()
    monkeypatch.setattr(s, "embed", lambda texts, kind: np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(s, "MIN_SCORE", 0.5)

    c = app.app.test_client()
    st = c.get("/api/stats").get_json()
    assert st["count"] == 1 and st["industries"] == ["製造"]

    js = c.get("/api/search?q=hello").get_json()
    assert "nodes" in js and "hidden" in js
    assert js["nodes"][0]["title"] == "A"


# ── API: 認証（全体/書き込み専用）──
def test_auth_full_and_write_only(env, monkeypatch):
    import app

    c = app.app.test_client()

    # 全体認証
    monkeypatch.setattr(app, "AUTH_PASSWORD", "secret")
    monkeypatch.setattr(app, "WRITE_PASSWORD", None)
    assert c.get("/api/stats").status_code == 401
    hdr = {"Authorization": "Basic " + base64.b64encode(b"user:secret").decode()}
    assert c.get("/api/stats", headers=hdr).status_code == 200

    # 書き込み専用認証（読みは自由・POSTのみ要認証）
    monkeypatch.setattr(app, "AUTH_PASSWORD", None)
    monkeypatch.setattr(app, "WRITE_PASSWORD", "w")
    assert c.get("/api/stats").status_code == 200
    assert c.post("/api/upload").status_code == 401


# ── API: 非同期アップロード → ジョブ完了 ──
def test_upload_async_job(env, monkeypatch):
    import app
    import jobs

    monkeypatch.setattr(env.search, "embed",
                        lambda texts, kind: np.array([[1.0, 0.0]], dtype=np.float32))

    c = app.app.test_client()
    data = {
        "files": (io.BytesIO("課題: x\n施策: y\n成果: z".encode()), "u.txt"),
        "industry": "",
    }
    r = c.post("/api/upload", data=data, content_type="multipart/form-data").get_json()
    assert "job_id" in r

    job = None
    for _ in range(80):  # 最大8秒待つ
        job = jobs.get(r["job_id"])
        if job and job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job and job["status"] == "done", job
    assert "u.txt" in job["added"]
