# 事例ファインダー（ローカル意味検索サーバー）

手元の事例（PPT / PDF）を、意味の近さで正確に探せるローカルアプリです。
Connected Papers のように「あなたの問題」を中心にしたグラフで関連事例を表示します。
**データは外部に送信しません。生成AI（LLM）は使いません。**
意味理解は BERT 系エンコーダ、画像中心の資料は OCR で扱います（モデルは初回のみDL）。

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

## セットアップ
```bash
python -m venv .venv && source .venv/bin/activate   # 任意
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
- **画面から**：上部の「＋ 事例を追加」でファイルを選び「取り込む」（業種は任意指定/自動推定）
- **CLIから**：`data/` に置いて `python ingest.py` を再実行（同名は上書き、`data/` から消した事例はDBからも自動削除）

## 運用（みんなで使う）
```bash
# 本番サーバー（waitress）で起動。社外秘ならBasic認証を有効化:
CASE_FINDER_PASSWORD=ひみつ python app.py
#   ユーザー名の既定は "user"（CASE_FINDER_USER で変更可）
```
- `CASE_FINDER_PASSWORD` 未設定だと**認証なしで公開**されます（同一LAN内の誰でも閲覧可）。社外秘の事例では設定を推奨。
- `waitress` が入っていれば自動で本番サーバー、無ければ警告付きで開発サーバーになります。

### 主な環境変数
| 変数 | 既定 | 説明 |
|---|---|---|
| `CASE_FINDER_MIN_SCORE` | `0.80` | これ未満の関連度は検索結果から除外 |
| `CASE_FINDER_OCR_ENGINE` | `easyocr` | `easyocr` / `tesseract` / `auto` |
| `CASE_FINDER_INDUSTRIES` | （内蔵リスト） | 業種候補をカンマ区切りで上書き |
| `CASE_FINDER_PASSWORD` | （なし） | Basic認証パスワード。設定すると認証必須 |
| `PORT` | `5000` | 待ち受けポート |

## 仕組み（生成AI不使用）
- 抽出: `PyMuPDF`（PDF）/ `python-pptx`（PPT）
- OCR: 既定は `EasyOCR`（Apache-2.0・商用可）、代替で `Tesseract`。テキスト層が乏しいページ/画像スライドを自動でOCR
- 意味ベクトル: `sentence-transformers` の `intfloat/multilingual-e5-small`
  （XLM-RoBERTa ＝ BERT 系エンコーダ。生成LLMではありません）
- 課題/施策/成果の仕分け・業種推定: 見出し検出 ＋ **BERT埋め込みのゼロショット分類**
  （各文/本文を埋め込み、ラベル代表文との類似度で割り当て）
- 関連度: 生コサインを閾値で足切りし、表示用に 0〜100% へ伸縮（無関係を出さない）
- 高速化: 起動時にベクトル行列をメモリ保持。取り込み時に自動更新（毎回の全読込を回避）
- 保存: SQLite（`cases.db`）。再起動しても再計算不要
- 検索: コサイン類似度（ベクトルは正規化済み）

別のモデルを使いたい場合は環境変数で指定できます:
```bash
CASE_FINDER_MODEL=intfloat/multilingual-e5-base python ingest.py
```

## 構成
| ファイル | 役割 |
|---|---|
| `ingest.py` | PPT/PDF を読み取り（必要ならOCR）→ 仕分け → ベクトル化 → DB登録 |
| `ocr.py` | 画像中心の資料を OCR で文字化（既定 EasyOCR / 代替 Tesseract） |
| `extract.py` | BERT埋め込みで「課題/施策/成果」に分類 |
| `search.py` | 埋め込み・DB・検索/グラフの中核ロジック |
| `app.py` | Flask サーバー（API + 画面配信） |
| `templates/index.html` | 画面（意味検索＋関係グラフ） |
| `data/` | 事例の PPT/PDF を置く場所 |
| `cases.db` | 事例テキスト＋ベクトルの保存先（自動生成） |

## 今後の案
- PaddleOCR など他エンジンの追加
- 画面からのアップロードで取り込み
- 分類しきい値やラベル代表文の調整UI
