# NDLOCR-Lite API (api-ndlocr-lite)

国立国会図書館（NDL）が公開している軽量OCRエンジン「[ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite)」を基盤とした、OpenAI互換（Mistral OCR形式）のOCR APIサーバーです。

## 概要
NDLOCR-Liteは、GPUを必要とせず、一般的なPC環境で高速に動作するOCRエンジンです。本プロジェクトは、この強力なエンジンを現代的なAIワークフローやエージェントから容易に利用できるよう、FastAPIを用いてAPI化したものです。

### 主な機能
- **OpenAI互換のスキーマ**: OpenAIやMistral AIのOCR APIに近いレスポンス形式を採用。
- **マルチモード対応**: 
  - `multipart/form-data` による画像ファイルの直接アップロード。
  - JSON形式によるBase64エンコード画像の送信。
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

---

## 開発者向け情報

### ローカルでのセットアップ (uvを使用)
[uv](https://github.com/astral-sh/uv)を使用して開発環境を構築します。

```bash
# 依存関係のインストール
uv sync

# テストの実行
PYTHONPATH=. uv run pytest
```

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

```bash
# 依存関係のインストール（初回のみ）
uv sync

# テストUIの起動
uv run streamlit run streamlit_app.py
```

ブラウザで `http://localhost:8501` にアクセスすると、以下の機能が利用できます：
- **同期OCR**: 画像をアップロードして即座に結果を取得
- **非同期ジョブ**: ジョブを作成し、ポーリングで結果を確認
- **ヘルスチェック**: サイドバーからAPIの稼働状況を確認

> **注意**: APIサーバー（Docker Compose）が起動している必要があります。デフォルトのAPI URLは `http://localhost:8001` です。

### セキュリティ制限
APIの安定稼働のため、デフォルトで以下の制限が設定されています。これらは環境変数で変更可能です。

- **MAX_IMAGE_SIZE**: アップロード可能な画像サイズ（初期値: 10MB）
- **MAX_BODY_SIZE**: リクエストボディの最大サイズ（初期値: 15MB）
- **MAX_PIXELS**: 画像の最大画素数（初期値: 100MP）

### プロジェクト構成
- `src/core/engine.py`: NDLOCR-Liteをラップした推論エンジン。
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
