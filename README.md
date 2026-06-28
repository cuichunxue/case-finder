# 事例ファインダー（ローカル意味検索サーバー）

手元の事例（PPT / PDF）を、意味の近さで正確に探せるローカルアプリです。
Connected Papers のように「あなたの問題」を中心にしたグラフで関連事例を表示します。
**データは外部に送信しません。生成AI（LLM）は使いません。**
意味理解は BERT 系エンコーダ、画像中心の資料は OCR で扱います（モデルは初回のみDL）。

検索は **チャンク密検索 ＋ BM25ハイブリッド ＋ クロスエンコーダ・リランカー** の
3段構成で、LLM-RAG 同等以上の精度を狙います（いずれも非生成・BERT系/語彙）。
精度は `bench.py`（Recall@k / MRR / nDCG）で測定できます。

さらに **Azure 生成AI（任意）** を有効にすると、検索上位を根拠にした
**要約・示唆（出典つき）**、クエリ拡張、Azure 埋め込みを上乗せできます。
未設定なら自動的に純ローカルへフォールバックします。

## できること
- `data/` に置いた PPT / PDF を読み取り、意味ベクトル化して登録
- **画面からアップロード**しても取り込める（CLI不要・その場で検索対象に）
- 画像中心のスライド・スキャンPDFは **OCR（既定 EasyOCR / 商用可）** で文字化
- **BERT 埋め込み**で各事例を「課題 / 施策 / 成果」に自動仕分け＋**業種を自動推定**（LLM不使用）
- 「困っていること」を文章で入力すると、関連度の高い事例をランキング表示（**業種で絞り込み可**）
- 関連度は閾値で足切りし、無関係な事例は出さない（**該当なしも正しく表示**）
- 事例同士のつながりをグラフで可視化／結果から元ファイルを開ける
- ローカルサーバーなので、同じネットワークの人みんなで使える
- 任意の **Basic認証**と**本番サーバー（waitress）**に対応
- 任意で **Azure 生成AI** を選択し、検索上位を根拠にした**要約・示唆（出典つき・逐次表示）**を生成
- 結果への 👍/👎 フィードバック記録、稼働メトリクス（`/api/metrics`）

## セットアップ

### A) Docker（推奨・一番かんたん）
```bash
mkdir -p store/data            # 事例(PPT/PDF)は store/data に置く
docker compose up -d --build   # http://localhost:5000
docker compose run --rm case-finder python ingest.py   # 取り込み（初回はモデルDL）
```
DB・モデルキャッシュはボリュームに永続化されます。社外秘なら compose の
`CASE_FINDER_PASSWORD` を設定してください。

### B) ローカル（venv）
```bash
./setup.sh                      # venv作成＋依存インストール
source .venv/bin/activate
# data/ に事例を置いてから
python ingest.py
./run.sh                        # = python app.py
```
手動で行う場合:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
OCR の既定エンジンは **EasyOCR**（Apache-2.0・商用利用可）です。`requirements.txt` に含まれ、
日本語モデルは初回の取り込み時に自動ダウンロードされます。追加導入は不要です。

OCR エンジンは環境変数で切り替えられます:
```bash
CASE_FINDER_OCR_ENGINE=easyocr    # 既定（高精度・pipのみ）
CASE_FINDER_OCR_ENGINE=tesseract  # 軽量。別途 Tesseract 本体の導入が必要
CASE_FINDER_OCR_ENGINE=auto       # EasyOCRがあれば優先、無ければTesseract
```
OCR エンジンが無くても、テキスト層のある PPT/PDF はそのまま取り込めます
（OCR は自動で有効/無効を判定します）。

## 使い方
1. `data/` フォルダに事例の PPT / PDF を入れる
2. 取り込み（初回はモデルを自動ダウンロード）
   ```bash
   python ingest.py
   ```
3. サーバー起動
   ```bash
   python app.py
   ```
4. ブラウザで `http://localhost:5000`
   - 同じネットワークの人は `http://<このPCのIP>:5000` でアクセス

事例の追加は2通り:
- **画面から**：上部の「＋ 事例を追加」でファイルを選び「取り込む」（業種は任意指定/自動推定）。
  取り込みは**バックグラウンドで実行**され、進捗（◯/◯件）が表示されます。
- **CLIから**：`data/` に置いて `python ingest.py` を再実行（同名は上書き、`data/` から消した事例はDBからも自動削除）

検索で「関連の高い事例が無い」場合は、**「弱い候補も表示」**から関連度のしきい値を下げて再検索できます。

## 運用（みんなで使う）
```bash
# 本番サーバー（waitress）で起動。社外秘ならBasic認証を有効化:
CASE_FINDER_PASSWORD=ひみつ python app.py            # 読み書きとも要認証
CASE_FINDER_WRITE_PASSWORD=ひみつ python app.py      # 読みは自由・取り込みのみ要認証
#   ユーザー名の既定は "user"（CASE_FINDER_USER で変更可）
```
- どちらも未設定だと**認証なしで公開**されます（同一LAN内の誰でも閲覧・取り込み可）。社外秘では設定を推奨。
- `waitress` が入っていれば自動で本番サーバー、無ければ警告付きで開発サーバーになります。
- 起動時に埋め込みモデルを先読み（warmup）するため、最初の検索の待ちが減ります。

## しきい値の校正（任意・精度の作り込み）
`MIN_SCORE` などの既定値は汎用の目安です。あなたの事例に合わせるには:
```bash
python ingest.py                       # 事例を登録
cp eval.sample.json eval.json          # 評価データを用意（query と正解事例）
python calibrate.py                    # 閾値スイープとスコア分布から推奨値を表示
```
出力された `CASE_FINDER_MIN_SCORE` 等を環境変数に設定して再起動すると反映されます。

### 主な環境変数
関連度は密検索・BM25・リランカーの判断を **0〜1 の統一スコア(rel)** にまとめ、
足切り・表示%・並び順をすべて rel で一貫させています（語彙ヒットやリランク上位が
密の閾値で消える問題を解消）。

| 変数 | 既定 | 説明 |
|---|---|---|
| `CASE_FINDER_MIN_REL` | `0.40` | これ未満は検索結果から除外（統一関連度） |
| `CASE_FINDER_WEAK_REL` | `0.15` | 「弱い候補も表示」で使う下限（統一関連度） |
| `CASE_FINDER_REL_FLOOR` / `_REL_CEIL` | `WEAK_FLOOR` / `0.92` | 密コサイン→関連度の伸縮範囲 |
| `CASE_FINDER_BM25_SAT` | `6.0` | BM25生スコアを0-1関連度に飽和変換する係数 |
| `CASE_FINDER_RERANK` / `_RERANKER` | `auto` / 日本語CE | リランカーの有効化とモデル |
| `CASE_FINDER_HYBRID` | `auto` | BM25ハイブリッドの有効化（`off`で密のみ） |
| `CASE_FINDER_ANN` / `_ANN_MIN` / `_ANN_K` | `auto` / `2000` / `200` | 大規模時にhnswlibで密検索を近似高速化 |
| `CASE_FINDER_CACHE_SIZE` / `_CACHE_TTL` | `256` / `300` | 結果・要約キャッシュのサイズ/秒 |
| `CASE_FINDER_OCR_ENGINE` | `easyocr` | `easyocr` / `tesseract` / `auto` |
| `CASE_FINDER_INDUSTRIES` | （内蔵リスト） | 業種候補をカンマ区切りで上書き |
| `CASE_FINDER_PASSWORD` | （なし） | 全体のBasic認証パスワード |
| `CASE_FINDER_WRITE_PASSWORD` | （なし） | 取り込みのみ要認証にする場合 |
| `PORT` | `5000` | 待ち受けポート |

## Azure 生成AI（任意の選択機能）
有効にすると、検索の上に生成AIの能力を上乗せできます。**未設定なら一切使われず、
完全ローカルのまま動作します。**

- **RAG要約**：検索上位の事例だけを根拠に、課題への示唆を日本語で生成（出典 [n] つき）
- **クエリ拡張(HyDE)**（任意）：仮想事例を生成して密検索の再現率を底上げ
- **Azure 埋め込み**（任意）：ローカル埋め込みの代わりに使用（取り込みと検索で揃える）

> ⚠ **プライバシー注意**：有効化すると、クエリや上位事例の本文が Azure に送信されます。
> 社外秘データを扱う場合は、契約・データ保持ポリシーを確認のうえご利用ください。

```bash
pip install openai
export CASE_FINDER_AZURE=on
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=<key>
export AZURE_OPENAI_CHAT_DEPLOYMENT=<chatデプロイ名>       # 要約・拡張
export AZURE_OPENAI_EMBED_DEPLOYMENT=<embeddingデプロイ名> # 任意
# 任意: クエリ拡張 / 埋め込みをAzureに
export CASE_FINDER_AZURE_EXPAND=on
export CASE_FINDER_EMBED_BACKEND=azure   # 使う場合は取り込みも同設定で再 ingest
python app.py
```
画面では、Azureが有効なときだけ「🧠 AIで要約（Azure）」トグルが現れます。
API は `GET /api/answer?q=...`。**検索結果(nodes/edges/hidden)も同梱して返す**ため、
AI要約ON時もフロントは1リクエストで完結します（二重検索なし）。Azure無効時は `answer:null`。

`/api/answer` は外部送信・課金を伴うため、**書き込みと同じ認証ゲート**で保護されます
（`CASE_FINDER_PASSWORD` か `CASE_FINDER_WRITE_PASSWORD` 設定時は要認証）。
Azure呼び出しには `CASE_FINDER_AZURE_TIMEOUT`（既定30秒）/ `CASE_FINDER_AZURE_RETRIES`（既定1）が効きます。
埋め込みバックエンドを切り替えた場合は、次元が変わるため必ず `python ingest.py` で再取り込みしてください
（不一致は検知してエラー表示します）。

## テスト
```bash
pip install pytest
pytest -q          # スタブ埋め込みで配線を検証（実モデル/OCR/Azure不要）
```

## 仕組み（生成AI不使用）
- 抽出: `PyMuPDF`（PDF）/ `python-pptx`（PPT）
- OCR: 既定は `EasyOCR`（Apache-2.0・商用可）、代替で `Tesseract`
- チャンク化: 本文を節単位に分割して各チャンクを埋め込み（長文・複数トピック対策）
- 意味ベクトル: `sentence-transformers` の `intfloat/multilingual-e5-small`
  （XLM-RoBERTa ＝ BERT 系エンコーダ。生成LLMではありません）
- **ハイブリッド検索**: 密検索（コサイン）＋ BM25（語彙一致）を RRF で融合
  → 固有名詞・型番・数値の取りこぼしを抑制
- **リランカー**: 上位候補を日本語クロスエンコーダで並べ替え（精度の最大レバー）
- **抜粋根拠**: ヒットの該当チャンクとクエリ語のハイライトで「なぜ近いか」を提示（生成なし）
- 課題/施策/成果の仕分け・業種推定: 見出し検出 ＋ BERT埋め込みのゼロショット分類
- 関連度: 生コサインを閾値で足切りし、表示用に 0〜100% へ伸縮（無関係を出さない）
- 高速化: 起動時にインデックスをメモリ保持＋モデルwarmup、結果/要約のTTLキャッシュ、
  大規模時は hnswlib による近似最近傍(ANN)。取り込み時に自動更新
- 保存: SQLite（`cases.db`、事例＋チャンク）。再起動しても再計算不要

### 精度を上げる（LLM同等以上を狙う設定）
既定は軽量モデルです。精度重視なら、より強い日本語埋め込み＋リランカーに差し替え:
```bash
# 例: 高精度な日本語埋め込み + 日本語リランカー（初回DLあり）
CASE_FINDER_MODEL=cl-nagoya/ruri-large \
CASE_FINDER_RERANKER=hotchpotch/japanese-reranker-cross-encoder-large-v1 \
python ingest.py && \
CASE_FINDER_MODEL=cl-nagoya/ruri-large python app.py
```
切替に関わる環境変数: `CASE_FINDER_MODEL` / `CASE_FINDER_RERANK`(auto|off) /
`CASE_FINDER_RERANKER` / `CASE_FINDER_HYBRID`(auto|off) / `CASE_FINDER_CHUNK_SIZE`。

### 精度の測定（A/B）
```bash
python bench.py                          # 現設定の Recall@k / MRR / nDCG
CASE_FINDER_HYBRID=off python bench.py   # 密のみと比較
CASE_FINDER_RERANK=off python bench.py   # リランカー無しと比較
```

## 構成
| ファイル | 役割 |
|---|---|
| `ingest.py` | PPT/PDF を読み取り（必要ならOCR）→ 仕分け → ベクトル化 → DB登録 |
| `ocr.py` | 画像中心の資料を OCR で文字化（既定 EasyOCR / 代替 Tesseract） |
| `extract.py` | BERT埋め込みで「課題/施策/成果」分類・業種推定 |
| `search.py` | チャンク密検索＋BM25ハイブリッド＋リランク・抜粋根拠の中核 |
| `rerank.py` | クロスエンコーダ・リランカー（未導入なら自動フォールバック） |
| `azure_ai.py` | Azure OpenAI連携（任意）：RAG要約・クエリ拡張・Azure埋め込み |
| `jobs.py` | 取り込みのバックグラウンド実行＋進捗＋モデルwarmup |
| `calibrate.py` | 評価データから閾値を校正するツール |
| `bench.py` | Recall@k / MRR / nDCG で検索精度を測定（A/B比較） |
| `app.py` | Flask サーバー（API + 画面配信 + 認証 + メトリクス/ログ） |
| `cache.py` / `metrics.py` | TTLキャッシュ / 稼働メトリクス |
| `Dockerfile` / `docker-compose.yml` | コンテナ配布。`./store` にDB・事例を永続化 |
| `setup.sh` / `run.sh` | ローカル(venv)導入・起動スクリプト |
| `templates/index.html` | 画面（意味検索＋関係グラフ＋根拠ハイライト＋アップロード） |
| `tests/` | スタブ埋め込みによるスモークテスト（12件） |
| `data/` | 事例の PPT/PDF を置く場所 |
| `cases.db` | 事例＋チャンク＋ベクトルの保存先（自動生成） |

## 今後の案
- PaddleOCR など他エンジンの追加
- 1ファイルに複数事例があるデッキのスライド/章単位チャンク化
- 画面からの事例の削除・一覧ブラウズ
