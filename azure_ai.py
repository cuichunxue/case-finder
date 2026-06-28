"""Azure OpenAI 連携（任意の選択機能）。

既定の検索は完全ローカル（純BERT）で動く。この機能を有効にしたときだけ、
Azure OpenAI を使って次を上乗せできる:

  - RAG要約 : 検索上位の事例を根拠に、課題への示唆を日本語で生成（出典つき）
  - クエリ拡張(HyDE) : クエリから仮想事例文を生成し、密検索の再現率を底上げ（任意）
  - 埋め込み : ローカルの代わりに Azure 埋め込みを使用（任意）

⚠ 有効化するとクエリや事例本文が Azure に送信されます（社外秘の取り扱いに注意）。
   未設定・SDK未導入なら available()=False となり、自動的に純BERTへフォールバックします。

必要な環境変数:
  CASE_FINDER_AZURE            = on            # 生成(要約)機能のマスタースイッチ
  AZURE_OPENAI_ENDPOINT        = https://<resource>.openai.azure.com
  AZURE_OPENAI_API_KEY         = <key>
  AZURE_OPENAI_API_VERSION     = 2024-06-01    # 既定
  AZURE_OPENAI_CHAT_DEPLOYMENT = <chatデプロイ名>      （要約・拡張用）
  AZURE_OPENAI_EMBED_DEPLOYMENT= <embeddingデプロイ名> （埋め込み用・任意）
  CASE_FINDER_AZURE_EXPAND     = on            # クエリ拡張を使う場合
  CASE_FINDER_EMBED_BACKEND    = azure         # 埋め込みをAzureにする場合（既定 local）
"""

from __future__ import annotations

import os

import numpy as np

import metrics

ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "")

AZURE_ON = os.environ.get("CASE_FINDER_AZURE", "off").lower() in ("on", "1", "true")
EXPAND_ON = os.environ.get("CASE_FINDER_AZURE_EXPAND", "off").lower() in ("on", "1", "true")
TIMEOUT = float(os.environ.get("CASE_FINDER_AZURE_TIMEOUT", "30"))
MAX_RETRIES = int(os.environ.get("CASE_FINDER_AZURE_RETRIES", "1"))

_state = {"client": None, "failed": False}


def _client():
    if _state["failed"]:
        return None
    if _state["client"] is not None:
        return _state["client"]
    if not (ENDPOINT and API_KEY):
        return None
    try:
        from openai import AzureOpenAI

        _state["client"] = AzureOpenAI(
            azure_endpoint=ENDPOINT, api_key=API_KEY, api_version=API_VERSION,
            timeout=TIMEOUT, max_retries=MAX_RETRIES,
        )
        return _state["client"]
    except Exception:  # noqa: BLE001
        _state["failed"] = True
        return None


def available() -> bool:
    """RAG要約（チャット）が使えるか。"""
    return AZURE_ON and bool(CHAT_DEPLOYMENT) and _client() is not None


def embeddings_available() -> bool:
    return bool(EMBED_DEPLOYMENT) and _client() is not None


def expand_enabled() -> bool:
    return EXPAND_ON and available()


def status() -> dict:
    return {
        "synth": available(),
        "expand": expand_enabled(),
        "embed": embeddings_available() and os.environ.get(
            "CASE_FINDER_EMBED_BACKEND", "local"
        ).lower() == "azure",
        "chat_deployment": CHAT_DEPLOYMENT if available() else "",
    }


# ──────────────────────────────────────────────────────────────
# 埋め込み（任意バックエンド）
# ──────────────────────────────────────────────────────────────
def embed(texts):
    """Azure 埋め込みで正規化済みベクトルを返す。e5用の接頭辞は付けない。"""
    client = _client()
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=list(texts))
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# クエリ拡張（HyDE）
# ──────────────────────────────────────────────────────────────
def expand_query(query: str) -> str:
    """クエリから仮想的な事例の説明文を生成し、密検索の手掛かりを増やす。"""
    client = _client()
    try:
        r = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            temperature=0.3,
            max_tokens=200,
            messages=[
                {"role": "system", "content":
                 "あなたは社内事例検索の補助です。ユーザーの困りごとに対応しそうな"
                 "架空の事例（課題・施策・成果）を3文程度で簡潔に書いてください。前置きは不要。"},
                {"role": "user", "content": query},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001
        return ""


# ──────────────────────────────────────────────────────────────
# RAG要約（根拠つき）
# ──────────────────────────────────────────────────────────────
def _format_cases(cases):
    lines, citations = [], []
    for i, c in enumerate(cases, 1):
        citations.append({"n": i, "title": c.get("title", ""), "source": c.get("source", ""),
                          "industry": c.get("industry", "")})
        body = "\n".join(filter(None, [
            f"課題: {c.get('problem','')}" if c.get("problem") else "",
            f"施策: {c.get('action','')}" if c.get("action") else "",
            f"成果: {c.get('result','')}" if c.get("result") else "",
            f"抜粋: {c.get('evidence','') or c.get('excerpt','')}",
        ]))
        lines.append(f"[{i}] {c.get('title','')}（{c.get('industry','') or '業種不明'}）\n{body}")
    return "\n\n".join(lines), citations


def synthesize(query: str, cases: list) -> dict:
    """検索上位の事例だけを根拠に、課題への示唆を生成する（出典[n]つき）。"""
    client = _client()
    context, citations = _format_cases(cases)
    system = (
        "あなたは社内の事例検索アシスタントです。次のルールに厳密に従ってください。\n"
        "1) 回答は必ず『提供された事例』のみを根拠にする。事例にない情報は創作しない。\n"
        "2) 各主張の文末に出典番号 [n] を付ける（複数可）。\n"
        "3) ユーザーの課題に対し、参考になる打ち手と理由を日本語で簡潔に（箇条書き可）。\n"
        "4) 関連が薄い場合は、その旨を正直に述べる。"
    )
    user = f"# ユーザーの課題\n{query}\n\n# 提供された事例\n{context}\n\n# 出力\n課題への示唆を、出典[n]つきでまとめてください。"
    r = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        temperature=0.2,
        max_tokens=700,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    answer = (r.choices[0].message.content or "").strip()
    metrics.incr("azure_synth")
    try:
        metrics.incr("azure_tokens", float(r.usage.total_tokens))
    except Exception:  # noqa: BLE001
        pass
    return {"answer": answer, "citations": citations, "model": CHAT_DEPLOYMENT}


def synthesize_stream(query: str, cases: list):
    """RAG要約をトークン単位でストリーミングする。最後に citations を返すジェネレータ。

    yield 形式: ("token", 文字列) を逐次、最後に ("citations", list) と ("done", model)。
    """
    client = _client()
    context, citations = _format_cases(cases)
    system = (
        "あなたは社内の事例検索アシスタントです。次のルールに厳密に従ってください。\n"
        "1) 回答は必ず『提供された事例』のみを根拠にする。事例にない情報は創作しない。\n"
        "2) 各主張の文末に出典番号 [n] を付ける（複数可）。\n"
        "3) ユーザーの課題に対し、参考になる打ち手と理由を日本語で簡潔に（箇条書き可）。\n"
        "4) 関連が薄い場合は、その旨を正直に述べる。"
    )
    user = f"# ユーザーの課題\n{query}\n\n# 提供された事例\n{context}\n\n# 出力\n課題への示唆を、出典[n]つきでまとめてください。"
    metrics.incr("azure_synth_stream")
    stream = client.chat.completions.create(
        model=CHAT_DEPLOYMENT, temperature=0.2, max_tokens=700, stream=True,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield ("token", chunk.choices[0].delta.content)
    yield ("citations", citations)
    yield ("done", CHAT_DEPLOYMENT)
