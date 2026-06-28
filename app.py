"""事例ファインダー ローカルサーバー。

    python app.py

起動後、同じネットワークの人は http://<あなたのIP>:5000 でアクセスできます。

任意で簡易Basic認証を有効化できます（社外秘の事例を扱う場合に推奨）:
    CASE_FINDER_PASSWORD=ひみつ python app.py
    （ユーザー名は既定 "user"、CASE_FINDER_USER で変更可）
"""

from __future__ import annotations

import os

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

import ingest
import jobs
import ocr
import search

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("CASE_FINDER_MAX_MB", "64")) * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 任意のBasic認証（読みと書きで分離可能）──
#   CASE_FINDER_PASSWORD       : 設定すると全アクセスに認証が必要
#   CASE_FINDER_WRITE_PASSWORD : 設定すると「読みは自由・書き(upload)のみ要認証」
AUTH_USER = os.environ.get("CASE_FINDER_USER", "user")
AUTH_PASSWORD = os.environ.get("CASE_FINDER_PASSWORD")
WRITE_PASSWORD = os.environ.get("CASE_FINDER_WRITE_PASSWORD")


def _creds_ok(passwords) -> bool:
    a = request.authorization
    return bool(a and a.username == AUTH_USER and a.password in passwords)


def _unauthorized():
    return Response(
        "認証が必要です", 401, {"WWW-Authenticate": 'Basic realm="case-finder"'}
    )


@app.before_request
def _require_auth():
    is_write = request.method in ("POST", "PUT", "DELETE")
    if is_write:
        # 書き込みは、全体パスワードか書き込み専用パスワードのいずれかで許可
        pws = {p for p in (AUTH_PASSWORD, WRITE_PASSWORD) if p}
        if pws and not _creds_ok(pws):
            return _unauthorized()
    else:
        if AUTH_PASSWORD and not _creds_ok({AUTH_PASSWORD}):
            return _unauthorized()
    return None


def _safe_filename(name: str) -> str:
    """日本語ファイル名は保ちつつ、パス区切りや '..' を排除する。"""
    name = os.path.basename(name or "").strip()
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        return ""
    return name


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    industry = request.args.get("industry", "").strip()
    top_k = int(request.args.get("k", 6))
    # loose=1 で「関連が弱い候補」も含める（閾値を WEAK_FLOOR まで下げる）
    min_score = search.WEAK_FLOOR if request.args.get("loose") else None
    return jsonify(search.search(q, top_k=top_k, industry=industry, min_score=min_score))


@app.route("/api/stats")
def api_stats():
    return jsonify(search.stats())


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """ブラウザからPPT/PDFを受け取り、保存後に取り込みをバックグラウンド実行する。

    ファイル保存だけ即時に行い、重い処理（OCR・埋め込み）はジョブ化して
    job_id を返す。進捗は /api/job/<id> でポーリングできる。
    """
    files = request.files.getlist("files")
    industry = request.form.get("industry", "").strip()
    if not files:
        return jsonify({"error": "ファイルがありません"}), 400

    os.makedirs(search.DATA_DIR, exist_ok=True)
    saved, rejected = [], []
    for f in files:
        name = _safe_filename(f.filename)
        if not name or not name.lower().endswith(ingest.SUPPORTED):
            rejected.append(f.filename or "(名前なし)")
            continue
        path = os.path.join(search.DATA_DIR, name)
        f.save(path)
        saved.append(path)

    if not saved:
        return jsonify({"error": "対応形式のファイルがありません", "rejected": rejected}), 400

    job_id = jobs.submit(saved, industry)
    return jsonify({"job_id": job_id, "queued": len(saved), "rejected": rejected})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "ジョブが見つかりません"}), 404
    return jsonify(job)


@app.route("/data/<path:filename>")
def data_file(filename):
    """検索結果から元の PPT/PDF を開けるようにする。"""
    return send_from_directory(os.path.join(BASE_DIR, "data"), filename)


def _run():
    port = int(os.environ.get("PORT", 5000))
    if not AUTH_PASSWORD and not WRITE_PASSWORD:
        print("※ 認証なしで公開します（読み書きとも自由）。社外秘なら CASE_FINDER_PASSWORD"
              " か CASE_FINDER_WRITE_PASSWORD の設定を推奨。")
    jobs.warmup()  # 埋め込みモデルを先読みして初回検索を速く
    print(f"\n事例ファインダーを起動します → http://0.0.0.0:{port}")
    print("同じネットワークの人は http://<このPCのIP>:%d で使えます。\n" % port)
    try:
        from waitress import serve  # 本番向けの安定サーバー

        serve(app, host="0.0.0.0", port=port, threads=int(os.environ.get("THREADS", "8")))
    except ImportError:
        print("（waitress未導入のため開発サーバーで起動。常用は pip install waitress を推奨）")
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    _run()
