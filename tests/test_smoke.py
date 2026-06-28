"""スモークテスト（スタブ埋め込み）。

実モデル/OCRなしで、抽出・チャンク検索・閾値/弱い候補・ハイブリッド・リランク・
抜粋根拠・prune・API・認証・非同期取り込みの配線を検証する。
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np

from conftest import store_one


def _q10(monkeypatch, s):
    """クエリ埋め込みを [1,0] 固定にし、関連度=コサインになるよう伸縮を無効化。"""
    monkeypatch.setattr(s, "embed", lambda texts, kind: np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(s, "REL_FLOOR", 0.0)
    monkeypatch.setattr(s, "REL_CEIL", 1.0)


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
    assert s.relevance(s.REL_FLOOR - 0.1) == 0.0


# ── 閾値・弱い候補(hidden)・loose ──
def test_search_threshold_and_hidden(env, monkeypatch):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 1.00)
    store_one(s, conn, "B", "b.txt", 0.85)
    store_one(s, conn, "C", "c.txt", 0.78)
    conn.close()
    s.invalidate_cache()

    _q10(monkeypatch, s)  # rel = cos
    monkeypatch.setattr(s, "MIN_REL", 0.80)
    monkeypatch.setattr(s, "WEAK_REL", 0.75)

    r = s.search("q", top_k=6)
    assert [n["title"] for n in r["nodes"]] == ["A", "B"]
    assert r["hidden"] == 1
    # 表示順と関連度%が一致（降順）すること
    assert r["nodes"][0]["relevance"] >= r["nodes"][1]["relevance"]

    r2 = s.search("q", top_k=6, min_rel=s.WEAK_REL)
    assert "C" in [n["title"] for n in r2["nodes"]]


# ── 業種フィルタ ──
def test_industry_filter(env, monkeypatch):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "M", "m.txt", 0.95, industry="製造")
    store_one(s, conn, "R", "r.txt", 0.95, industry="小売・EC")
    conn.close()
    s.invalidate_cache()
    _q10(monkeypatch, s)
    r = s.search("q", top_k=6, industry="製造", min_rel=0.0)
    assert [n["title"] for n in r["nodes"]] == ["M"]
    assert s.list_industries() == ["小売・EC", "製造"]


# ── prune（事例とチャンクの両方が消える）──
def test_prune(env):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "X", "data/gone.txt", 1.0)
    assert "data/gone.txt" in s.all_sources(conn)
    chunks_before = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    assert chunks_before == 1
    s.delete_sources(conn, {"data/gone.txt"})
    assert "data/gone.txt" not in s.all_sources(conn)
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 0
    conn.close()


# ── キャッシュ ──
def test_cache_invalidate(env):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 1.0)
    conn.close()
    s.invalidate_cache()
    assert len(s.get_index()["meta"]) == 1


# ── 抜粋根拠とハイライト用語 ──
def test_matched_spans(env):
    s = env.search
    spans = s.matched_spans("属人化のベテラン頼み", "ベテラン頼みで属人化している")
    # 最大長で重複排除されるため "属人化" と "ベテラン頼み" のように現れる
    assert any("属人化" in x for x in spans)
    assert any("ベテラン" in x for x in spans)


# ── ハイブリッド：完全一致語で1件に寄せる ──
def test_hybrid_lexical_match(env, monkeypatch):
    s = env.search
    monkeypatch.setattr(s, "HYBRID", "auto")  # BM25 を有効化
    conn = s.connect()
    s.init_db(conn)
    for title, src, text in [
        ("型番の不具合", "p.txt", "型番ABC123 の不具合と対策"),
        ("別の話題", "o.txt", "まったく無関係な内容"),
    ]:
        vec = s.embed([text], "passage")[0]
        s.store_case(conn, title, src, text, "e", "", {}, [(text, vec)])
    conn.close()
    s.invalidate_cache()
    # クエリ "ABC123" は密では弱いが BM25 の完全一致で p.txt が先頭に来る
    assert s.ranked_sources("ABC123", limit=2)[0] == "p.txt"


# ── リランカー：順序を入れ替える ──
def test_reranker_reorders(env, monkeypatch):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 0.90)
    store_one(s, conn, "B", "b.txt", 0.85)
    conn.close()
    s.invalidate_cache()
    _q10(monkeypatch, s)
    monkeypatch.setattr(s, "MIN_REL", 0.4)
    # 候補（融合順 A,B）に対し B を高評価にするリランカーを差し込む
    monkeypatch.setattr(env.rerank, "available", lambda: True)
    monkeypatch.setattr(env.rerank, "rerank", lambda q, texts: [0.0, 1.0])
    r = s.search("q", top_k=6)
    # リランカーの判断（B優位）が並び順・関連度に反映される
    assert [n["title"] for n in r["nodes"]] == ["B", "A"]
    assert r["nodes"][0]["relevance"] >= r["nodes"][1]["relevance"]


# ── API: stats / search（evidence・matched付き）──
def test_api_stats_and_search(env, monkeypatch):
    import app

    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 1.0, industry="製造", text="属人化を解消した")
    conn.close()
    s.invalidate_cache()
    _q10(monkeypatch, s)
    monkeypatch.setattr(s, "MIN_REL", 0.5)

    c = app.app.test_client()
    st = c.get("/api/stats").get_json()
    assert st["count"] == 1 and st["industries"] == ["製造"]
    assert "hybrid" in st and "reranker" in st

    js = c.get("/api/search?q=属人化").get_json()
    assert "nodes" in js and "hidden" in js
    n0 = js["nodes"][0]
    assert n0["title"] == "A"
    assert "evidence" in n0 and "matched" in n0


# ── API: 認証（全体/書き込み専用）──
def test_auth_full_and_write_only(env, monkeypatch):
    import app

    c = app.app.test_client()

    monkeypatch.setattr(app, "AUTH_PASSWORD", "secret")
    monkeypatch.setattr(app, "WRITE_PASSWORD", None)
    assert c.get("/api/stats").status_code == 401
    hdr = {"Authorization": "Basic " + base64.b64encode(b"user:secret").decode()}
    assert c.get("/api/stats", headers=hdr).status_code == 200

    monkeypatch.setattr(app, "AUTH_PASSWORD", None)
    monkeypatch.setattr(app, "WRITE_PASSWORD", "w")
    assert c.get("/api/stats").status_code == 200
    assert c.post("/api/upload").status_code == 401


# ── API: 非同期アップロード → ジョブ完了 ──
# ── Azure: 未設定なら無効＋/api/answer はフォールバック ──
def test_azure_disabled_by_default(env):
    import azure_ai

    assert azure_ai.available() is False
    assert search_stats_azure_off(env)


def search_stats_azure_off(env):
    az = env.search.stats()["azure"]
    return az["synth"] is False and az["embed"] is False


def test_api_answer_fallback(env, monkeypatch):
    import app

    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 1.0)
    conn.close()
    s.invalidate_cache()
    _q10(monkeypatch, s)
    monkeypatch.setattr(s, "MIN_REL", 0.5)

    c = app.app.test_client()
    j = c.get("/api/answer?q=hello").get_json()
    assert j["answer"] is None        # Azure未設定 → 生成なし
    assert j["nodes"][0]["title"] == "A"  # 検索結果は通常どおり返る


# ── Azure: スタブ要約が /api/answer から返る ──
def test_api_answer_with_stub(env, monkeypatch):
    import app
    import azure_ai

    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 1.0, text="属人化を解消した")
    conn.close()
    s.invalidate_cache()
    _q10(monkeypatch, s)
    monkeypatch.setattr(s, "MIN_REL", 0.5)

    monkeypatch.setattr(azure_ai, "available", lambda: True)
    monkeypatch.setattr(azure_ai, "synthesize",
                        lambda q, cases: {"answer": "メンター制度が有効です [1]",
                                          "citations": [{"n": 1, "title": cases[0]["title"],
                                                         "source": cases[0]["source"], "industry": ""}],
                                          "model": "gpt-stub"})
    c = app.app.test_client()
    j = c.get("/api/answer?q=属人化").get_json()
    assert j["answer"] == "メンター制度が有効です [1]"
    assert j["citations"][0]["title"] == "A"
    assert j["model"] == "gpt-stub"


def test_upload_async_job(env, monkeypatch):
    import app
    import jobs

    # 同期モードで決定的に処理（バックグラウンドスレッドの非決定性を排除）
    monkeypatch.setenv("CASE_FINDER_SYNC_JOBS", "1")

    c = app.app.test_client()
    data = {
        "files": (io.BytesIO("課題: x\n施策: y\n成果: z".encode()), "u.txt"),
        "industry": "",
    }
    r = c.post("/api/upload", data=data, content_type="multipart/form-data").get_json()
    assert "job_id" in r
    job = jobs.get(r["job_id"])
    assert job and job["status"] == "done", job
    assert "u.txt" in job["added"]
