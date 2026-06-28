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
    # 外部送信・課金を伴う /api/answer も書き込み相当として保護される
    assert c.get("/api/answer?q=x").status_code == 401


# ── 埋め込み次元の不一致をガードしてerror JSONを返す ──
def test_embed_dim_mismatch_guarded(env, monkeypatch):
    import app

    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 1.0)  # 2次元で保存
    conn.close()
    s.invalidate_cache()
    # クエリだけ3次元を返すよう壊す → 検知してerror
    monkeypatch.setattr(s, "embed", lambda texts, kind: np.zeros((1, 3), dtype=np.float32))
    r = c = app.app.test_client().get("/api/search?q=hello")
    assert r.status_code == 400
    assert "error" in r.get_json()


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


# ── 結果キャッシュ：同一クエリは再計算せず同一オブジェクトを返す ──
def test_result_cache(env, monkeypatch):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 1.0)
    conn.close()
    s.invalidate_cache()
    _q10(monkeypatch, s)
    monkeypatch.setattr(s, "MIN_REL", 0.5)

    r1 = s.search("q", top_k=6)
    r2 = s.search("q", top_k=6)
    assert r1 is r2  # キャッシュヒット（同一オブジェクト）
    s.invalidate_cache()
    r3 = s.search("q", top_k=6)
    assert r3 is not r1  # 失効後は作り直し


# ── ANN（hnswlib）の結果が総当たりと一致（小規模・閾値を下げて検証）──
def test_ann_matches_bruteforce(env, monkeypatch):
    import importlib.util

    import pytest
    if importlib.util.find_spec("hnswlib") is None:
        pytest.skip("hnswlib 未導入")
    s = env.search

    def build(cases):
        conn = s.connect()
        s.init_db(conn)
        for t, src, cos in cases:
            store_one(s, conn, t, src, cos)
        conn.close()
        s.invalidate_cache()

    cases = [("A", "a.txt", 0.95), ("B", "b.txt", 0.88), ("C", "c.txt", 0.83)]
    _q10(monkeypatch, s)
    monkeypatch.setattr(s, "MIN_REL", 0.0)

    build(cases)
    monkeypatch.setattr(s, "ANN", "off")
    s.invalidate_cache()
    brute = [n["title"] for n in s.search("q", top_k=6)["nodes"]]

    monkeypatch.setattr(s, "ANN", "on")
    monkeypatch.setattr(s, "ANN_MIN", 1)
    monkeypatch.setattr(s, "ANN_K", 10)
    s.invalidate_cache()
    assert s.get_index()["ann"] is not None      # ANN索引が構築された
    ann = [n["title"] for n in s.search("q", top_k=6)["nodes"]]
    assert ann == brute


# ── 要約キャッシュ：同一クエリは synthesize を1回だけ呼ぶ ──
def test_answer_cache(env, monkeypatch):
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

    calls = {"n": 0}

    def fake_synth(q, cases):
        calls["n"] += 1
        return {"answer": "回答 [1]", "citations": [], "model": "stub"}

    monkeypatch.setattr(azure_ai, "available", lambda: True)
    monkeypatch.setattr(azure_ai, "synthesize", fake_synth)
    c = app.app.test_client()
    a1 = c.get("/api/answer?q=属人化").get_json()
    a2 = c.get("/api/answer?q=属人化").get_json()
    assert a1["answer"] == "回答 [1]" and a2["answer"] == "回答 [1]"
    assert calls["n"] == 1  # 2回目はキャッシュ


# ── 観測性: /api/metrics にカウンタとキャッシュ統計が出る ──
def test_metrics_endpoint(env, monkeypatch):
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
    c.get("/api/search?q=hello")
    m = c.get("/api/metrics").get_json()
    assert m["counters"].get("searches", 0) >= 1
    assert "result" in m["cache"] and "answer" in m["cache"]


# ── フィードバック: 👍がDBに記録される ──
def test_feedback_recorded(env):
    import app

    s = env.search
    conn = s.connect()
    s.init_db(conn)
    conn.close()
    c = app.app.test_client()
    r = c.post("/api/feedback", data={"q": "離職", "case_id": "1", "vote": "up"})
    assert r.get_json()["ok"] is True
    conn = s.connect()
    row = conn.execute("SELECT query, vote FROM feedback").fetchone()
    conn.close()
    assert row["query"] == "離職" and row["vote"] == "up"


# ── ANN×業種: 業種フィルタ時は ANN を使わず総当たり ──
def test_ann_bypassed_with_industry(env, monkeypatch):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    store_one(s, conn, "A", "a.txt", 1.0, industry="製造")
    conn.close()
    s.invalidate_cache()
    _q10(monkeypatch, s)

    seen = {}
    orig = s._rank_cases
    monkeypatch.setattr(s, "_rank_cases",
                        lambda ix, q, use_ann=True: seen.update(use_ann=use_ann) or orig(ix, q, use_ann))
    s.search("q", industry="製造", min_rel=0.0)
    assert seen["use_ann"] is False
    s.search("q", min_rel=0.0)
    assert seen["use_ann"] is True


# ── ストリーミング要約: SSEで results→token→done が流れる ──
def test_answer_stream_stub(env, monkeypatch):
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
    monkeypatch.setattr(azure_ai, "synthesize_stream",
                        lambda q, cases: iter([("token", "あ"), ("token", "い"),
                                               ("citations", [{"n": 1, "title": "A", "source": "a.txt", "industry": ""}]),
                                               ("done", "stub")]))
    c = app.app.test_client()
    text = c.get("/api/answer_stream?q=属人化").get_data(as_text=True)
    assert '"type": "results"' in text
    assert "あ" in text and "い" in text
    assert '"type": "done"' in text


# ── SQLite: WAL とロック待ちが有効 ──
def test_sqlite_wal_enabled(env):
    s = env.search
    conn = s.connect()
    s.init_db(conn)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
    conn.close()


# ── /healthz は認証不要で疎通する ──
def test_healthz_public(env, monkeypatch):
    import app

    monkeypatch.setattr(app, "AUTH_PASSWORD", "secret")  # 全体認証ありでも
    c = app.app.test_client()
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


# ── デバイス解決：明示cpu/autoともに妥当な値 ──
def test_device_resolve(monkeypatch):
    import device

    monkeypatch.setenv("CASE_FINDER_DEVICE", "cpu")
    device.resolve.cache_clear()
    assert device.resolve() == "cpu"
    assert device.use_gpu() is False
    monkeypatch.setenv("CASE_FINDER_DEVICE", "auto")
    device.resolve.cache_clear()
    assert device.resolve() in ("cpu", "cuda", "mps")
    device.resolve.cache_clear()


# ── トピック地図：全事例がクラスタ・座標を持ち、キーワードが出る ──
def test_topic_map(env):
    import topics

    s = env.search
    conn = s.connect()
    s.init_db(conn)
    # 2グループ（離職系 / EC系）を別方向のベクトルで作る
    import math

    import numpy as np
    grpA = [("若手の離職を抑制", "a1.txt", "若手の離職と定着の課題"),
            ("離職率の改善", "a2.txt", "離職を防ぐ定着施策")]
    grpB = [("ECの離脱を改善", "b1.txt", "ECサイトの離脱とUI改善"),
            ("カート離脱対策", "b2.txt", "離脱率をUIで改善")]
    for i, (t, src, text) in enumerate(grpA):
        v = np.array([1.0, 0.0], dtype=np.float32)
        s.store_case(conn, t, src, text, "e", "", {}, [(text, v)])
    for t, src, text in grpB:
        v = np.array([0.0, 1.0], dtype=np.float32)
        s.store_case(conn, t, src, text, "e", "", {}, [(text, v)])
    conn.close()
    s.invalidate_cache()

    m = topics.build_map(s.get_index(), k=2)
    assert len(m["cases"]) == 4
    assert all("cluster" in c and "x" in c and "y" in c for c in m["cases"])
    assert len(m["clusters"]) == 2
    # 同方向ベクトルの2件は同じクラスタに入る
    cl = {c["title"]: c["cluster"] for c in m["cases"]}
    assert cl["若手の離職を抑制"] == cl["離職率の改善"]
    assert cl["ECの離脱を改善"] == cl["カート離脱対策"]
    assert any(cl_["keywords"] for cl_ in m["clusters"])  # キーワードが付く


# ── /api/map エンドポイント ──
def test_api_map(env):
    import app

    s = env.search
    conn = s.connect()
    s.init_db(conn)
    import numpy as np
    s.store_case(conn, "A", "a.txt", "離職の課題", "e", "", {}, [("離職の課題", np.array([1.0, 0.0], dtype=np.float32))])
    conn.close()
    s.invalidate_cache()
    j = app.app.test_client().get("/api/map").get_json()
    assert "clusters" in j and "cases" in j and len(j["cases"]) == 1


# ── 評価指標（純関数）──
def test_eval_metrics():
    import bench

    # 1位が正解(grade2)、3位に正解(grade1)
    grades = [2, 0, 1, 0, 0]
    rel = [2, 1]
    qm = bench.query_metrics(grades, rel)
    assert qm["hit"] == 1.0
    assert qm["mrr"] == 1.0
    assert qm["recall@1"] == 0.5 and qm["recall@3"] == 1.0
    assert 0.9 < qm["ndcg@5"] <= 1.0  # 理想に近い
    mean, lo, hi = bench.bootstrap_ci([1.0, 1.0, 0.0, 1.0])
    assert lo <= mean <= hi


# ── チャンクのオーバーラップ ──
def test_chunk_overlap(env):
    import ingest

    text = "\n".join(f"段落{i}の内容です。" * 8 for i in range(4))
    chunks = ingest.chunk_text(text, size=80, overlap=20)
    assert len(chunks) >= 2
    # 2つ目以降の先頭に直前チャンク末尾が含まれる
    tail = chunks[0][-20:]
    assert tail[:6] in chunks[1]


# ── トークナイザ：既定(char)はbigram ──
def test_tokenizer_char(env, monkeypatch):
    s = env.search
    monkeypatch.setattr(s, "TOKENIZER", "char")
    toks = s.tokenize("離職対策ABC")
    assert "離職" in toks and "abc" in toks


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
