# NDLOCR-Lite API (api-ndlocr-lite)

国立国会図書館（NDL）が公開している軽量OCRエンジン「[ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite)」を基盤とした、OpenAI互換（Mistral OCR形式）のOCR APIサーバーです。

## 概要
NDLOCR-Liteは、GPUを必要とせず、一般的なPC環境で高速に動作するOCRエンジンです。本プロジェクトは、この強力なエンジンを現代的なAIワークフローやエージェントから容易に利用できるよう、FastAPIを用いてAPI化したものです。

### 主な機能
- **NDLOCR-Lite ver.2 (`2.0.0`) 対応**: 英文・手書き文字の認識精度向上、句読点周辺の不要空白除去、重複検出領域の抑制に対応。
- **オプション機能（環境変数で有効化）**:
  - **異体字包摂 (`ENABLE_CHARACTER_INCLUSION`)**: 旧字体などの異体字を新字体へ変換して出力。
  - **構造化 Markdown 出力 (`ENABLE_RICH_MARKDOWN`)**: 見出しや段落結合、表埋め込みを反映した Markdown を出力。
  - **表構造化 (`ENABLE_TABLE_STRUCTURE`)**: TableCenterNet により表のセル構造を推定し、HTML / Markdown 形式で出力。
- **OpenAI互換の Vision API サポート**: `/v1/chat/completions` に画像を送信することで、OpenAI互換のレスポンスでOCR結果を取得可能。
- **OpenAI互換のスキーマ**: OpenAIやMistral AIのOCR APIに近いレスポンス形式を採用。
- **マルチモード対応**: 
  - `multipart/form-data` による画像ファイルの直接アップロード。
  - JSON形式によるBase64エンコード画像の送信。
- **安全性向上**: `defusedxml` の採用により、XXE（XML外部実体参照）攻撃やBillion Laughs攻撃などの脆弱性から保護。
- **パフォーマンス最適化**: OCRループ内の冗長なXMLツリー解析をキャッシュ化し、推論処理を高速化。
- **非同期ジョブ管理**: 時間のかかるOCR処理をバックグラウンドで実行し、ジョブIDでステータスを確認可能（`/v1/ocr/jobs`）。
- **高速な推論開始**: Lifespan機能により、サーバー起動時にモデルを一度だけロードするため、リクエストごとの遅延がありません。
- **Docker対応**: Docker Composeにより、環境構築なしですぐに利用可能です。

---

## クイックスタート

### 1. 準備
リポジトリをクローンし、サブモジュールを初期化します。

```bash
git clone --recursive https://github.com/chottokun/api-ndlocr-lite.git
cd api-ndlocr-lite
```

### 2. Docker Compose で実行（推奨）
Dockerを使用すると、複雑な依存関係のセットアップが不要です。

```bash
docker-compose up --build
```
サーバーが起動すると、`http://localhost:8000` でAPIが利用可能になります。

### 3. 環境設定
セキュリティ制限などの設定は環境変数で行うことができます。詳細は `.env.sample` を参照してください。

```bash
cp .env.sample .env
# 必要に応じて .env を編集
```

---

## API の利用方法

### 健康状態の確認 (Health Check)
```bash
curl http://localhost:8000/health
```

### 同期OCR（ファイルを直接アップロード）
```bash
curl -X POST http://localhost:8000/v1/ocr \
  -F "file=@/path/to/your/image.jpg"
```

### 非同期OCR（ジョブとして実行）
処理に時間がかかる画像や、大量の画像をバッチ処理する場合に適しています。

1. **ジョブの作成**
   ```bash
   curl -X POST http://localhost:8000/v1/ocr/jobs \
     -F "file=@/path/to/your/image.jpg"
   ```
   レスポンスに含まれる `job_id` をメモしてください。

2. **ステータスと結果の確認**
   ```bash
   curl http://localhost:8000/v1/ocr/jobs/{job_id}
   ```

### OpenAI互換 Vision API (Chat Completions)
OpenAI の Chat Completions API スキーマと互換性のあるエンドポイントです。LLMエージェントや外部ワークフローから容易に呼び出すことができます。

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ndlocr-lite",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "OCR this image"},
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,...(画像Base64データ)..."}
          }
        ]
      }
    ]
  }'
```

レスポンスの `message.content` には抽出されたプレーンテキスト（マークダウン形式）が含まれ、さらに `tool_calls` 内の `get_ocr_details` 関数の引数として、座標情報や信頼度などの詳細なメタデータが含まれます。

### OCR結果の拡張フィールド
認識結果の各テキスト行（`lines`）オブジェクトには、以下の拡張フィールドが含まれます：
- `isVertical`: `true`/`false` 形式の文字列。行のテキストが縦書きかどうかを判定します。
- `isTextline`: 常に `"true"`。テキスト行であることを示します。
- `class_index`: レイアウト種別を表すクラスインデックス（整数）。

また、`ENABLE_TABLE_STRUCTURE=true` を設定した場合、各ページ（`pages`）オブジェクトに `tables` フィールドが含まれます：
- `table_index`: ページ内の表番号（1始まり）。
- `layout_bbox`: 表領域のバウンディングボックス `[xmin, ymin, xmax, ymax]`。
- `html`: 推定された表の HTML 文字列（`rowspan` / `colspan` 対応）。
- `markdown`: 推定された表の Markdown テーブル文字列。
- `num_cells`: 検出されたセル数。

---

## 開発者向け情報

### ローカルでのセットアップ (uvを使用)
[uv](https://github.com/astral-sh/uv)を使用して開発環境を構築します。

```bash
# 依存関係のインストール
uv sync

# テストの実行（ユニット・統合・脆弱性回帰テスト）
PYTHONPATH=. uv run pytest

# 静的セキュリティ解析 (Bandit) および依存パッケージ脆弱性スキャン (pip-audit)
uv run bandit -r src/ -ll
uv run pip-audit --desc off
```

GitHub Actions（`.github/workflows/ci.yml`）では、`main` ブランチへのプッシュ・PR および週次スケジュール実行時に、自動テスト・Bandit・pip-audit・Docker ビルド・Trivy コンテナ脆弱性スキャン・結合スモークテストが実行されます。

### 負荷テストの実行
Locustを使用した負荷テストが可能です。

```bash
# 負荷テストの実行（サーバーの起動からテスト完了まで自動で行います）
./run_load_tests.sh
```

テスト項目:
- `/health` への定期的なアクセス。
- `/v1/ocr` への同期OCRリクエスト。
- `/v1/ocr/jobs` を使用した非同期ジョブの作成とポーリング。

### テスト用UI (Streamlit)
Streamlitを使用した簡易的なテスト用UIを内蔵しています。ブラウザ上で画像をアップロードし、OCR結果をインタラクティブに確認できます。

UIを起動するには、Dockerとは別にローカル環境で立ち上げることを推奨します（.venvの権限問題を避けるため）。

```bash
# 依存関係のインストール（初回のみ）
uv sync

# テストUIの起動（バックグラウンド等で）
uv run streamlit run streamlit_app.py --server.port 8501
```

ブラウザで `http://localhost:8501` にアクセスすると、以下の機能が利用できます：
- **同期OCR**: 画像をアップロードして即座に結果を取得
- **非同期ジョブ**: ジョブを作成し、ポーリングで結果を確認
- **ヘルスチェック**: サイドバーからAPIの稼働状況を確認

> **注意**: APIサーバー（Docker Compose）が起動している必要があります。デフォルトのAPI URLは `http://localhost:8001` です。

### OCR オプション設定（デフォルト値とリクエスト単位の指定）
`.env` ファイルまたは環境変数に設定した値がサーバーのデフォルト動作になります（ローカル起動時も `.env` が自動で読み込まれます。すべて初期値は `false` です）。

- **ENABLE_TCY**: 縦中横（TCY）認識を有効化（`true` / `false`）
- **ENABLE_CHARACTER_INCLUSION**: 異体字包摂（旧字体→新字体変換）を有効化（`true` / `false`）
- **ENABLE_RICH_MARKDOWN**: レイアウト解析に基づく構造化 Markdown 出力を有効化（`true` / `false`）
- **ENABLE_TABLE_STRUCTURE**: TableCenterNet による表構造化（HTML / Markdown 表出力）を有効化（`true` / `false`）

`character_inclusion` / `rich_markdown` / `table_structure` は、API コール時にリクエスト単位で上書きすることもできます（優先順位: **クエリパラメータ > JSON ボディ > `.env` / 環境変数**）。

```bash
# 例1: クエリパラメータでリクエスト単位に有効化（multipart / JSON どちらでも利用可能）
curl -X POST "http://localhost:8000/v1/ocr?rich_markdown=true&table_structure=true&character_inclusion=true" \
  -F "file=@/path/to/your/image.jpg"

# 例2: JSON ボディの任意フィールドで指定（/v1/ocr, /v1/ocr/jobs, /v1/chat/completions に対応）
curl -X POST http://localhost:8000/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "rich_markdown": true,
    "table_structure": true,
    "character_inclusion": false
  }'
```

### セキュリティ制限
APIの安定稼働のため、デフォルトで以下の制限が設定されています。これらは環境変数で変更可能です。

- **MAX_IMAGE_SIZE**: アップロード可能な画像サイズ（初期値: 10MB）
- **MAX_BODY_SIZE**: リクエストボディの最大サイズ（初期値: 15MB）
- **MAX_PIXELS**: 画像の最大画素数（初期値: 100MP）
- **XML外部エンティティ保護**: `defusedxml` により、悪意のあるXML入力を安全に処理します。

### プロジェクト構成
- `docs/architecture.md`: [システムの設計とデータフローの解説](docs/architecture.md)
- `src/core/engine.py`: NDLOCR-Liteをラップした推論エンジン。`defusedxml` による安全なXMLパース処理を含む。
- `src/api/main.py`: FastAPIによるAPIエンドポイントとジョブ管理。
- `src/schemas/ocr.py`: Pydanticによるリクエスト・レスポンスのスキーマ定義。
- `streamlit_app.py`: Streamlitによるテスト用UIアプリケーション。
- `extern/ndlocr-lite`: 本体のOCRエンジン（Git Submodule）。


---

## ライセンス・引用
- **本APIコード**: MIT License（またはリポジトリの設定に準ずる）
- **OCRエンジン (NDLOCR-Lite)**: [CC BY 4.0](https://github.com/ndl-lab/ndlocr-lite/blob/main/LICENCE) (国立国会図書館)
- **依存ライブラリ**: 各ライブラリのライセンスに基づきます。

NDLOCR-Liteの詳細な技術情報やモデルの著作権については、公式リポジトリを参照してください。
[https://github.com/ndl-lab/ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite)
