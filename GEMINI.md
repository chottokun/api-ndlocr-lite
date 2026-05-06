# GEMINI.md - api-ndlocr-lite

このドキュメントは、`api-ndlocr-lite` プロジェクトの開発、保守、および拡張に関する重要な指針と要件をまとめたものです。

## 1. プロジェクト概要
国立国会図書館 (NDL) の `ndlocr-lite` をベースにした、OpenAI/Mistral OCR互換スキーマを持つOCR APIサーバー。
- **目的**: 軽量・高速なOCRエンジンを現代的なAIワークフロー（エージェント、LLM統合等）から容易に利用可能にする。
- **特徴**: GPU不要（ONNXRuntime CPU）、非同期ジョブ管理、OpenAI互換レスポンス。
- **最新機能 (v1.2.1準拠)**:
  - 24px 高解像度認識モデルの採用。
  - 縦中横 (TCY) サポートによる新聞・雑誌資料の認識精度向上。
  - 長文（98文字以上）の自動分割認識ロジック。
  - ページ内の縦書き/横書き判定に基づくテキスト順序の自動調整。
  - `class_index` によるレイアウト情報の提供。

## 2. 技術スタック & 環境要件
- **言語**: Python 3.13+
- **主要ライブラリ**:
  - **API**: FastAPI, Uvicorn, Pydantic (v2)
  - **推論**: ONNXRuntime (CPU推奨、GPU対応可), NumPy, OpenCV (headless)
  - **画像処理**: Pillow (PIL)
  - **セキュリティ**: `defusedxml` (XML外部実体参照攻撃防止)
  - **管理**: `uv` (パッケージマネージャー), Docker / Docker Compose
- **サブモジュール**: `extern/ndlocr-lite` に本体エンジンとモデルが配置されている。

## 3. ディレクトリ構成
- `src/api/`: APIエンドポイント定義 (`main.py`)、ジョブ管理、依存関係。
- `src/core/`: OCRエンジンラッパー (`engine.py`)。サブモジュールとの橋渡し。
- `src/schemas/`: Pydanticモデル (`ocr.py`)。OpenAI互換スキーマ。
- `extern/ndlocr-lite/`: NDL OCR本体 (Git Submodule)。
- `tests/`: ユニットテスト、統合テスト、負荷テスト (`locustfile.py`)。
- `docs/`: アーキテクチャドキュメント、負荷テストレポート。

## 4. アーキテクチャ原則
- **Lifespan管理**: `FastAPI` の lifespan を使用し、起動時にモデルを一度だけロード (`app.state.engine`)。
- **ステートレス性**: API自体はステートレスだが、非同期ジョブ用に `InMemoryJobStore` を持つ（将来的にRedis等への置換を想定）。
- **非同期処理**: `BackgroundTasks` を使用して時間のかかるOCR処理を実行。
- **エンジン設計**: `NDLOCREngine` は `ThreadPoolExecutor` を内蔵し、行単位の認識を並列化。

## 5. コーディング規約 & セキュリティ
- **型ヒント**: 全ての関数とメソッドに厳密な型ヒントを付与する。
- **XMLセキュリティ**: XMLのパースには必ず `defusedxml` を使用し、`lxml` や標準の `xml.etree` を直接使用しない。
- **バリデーション**:
  - `MAX_IMAGE_SIZE` (デフォルト 10MB)
  - `MAX_BODY_SIZE` (デフォルト 15MB)
  - `MAX_PIXELS` (デフォルト 100MP)
  これらは環境変数で調整可能。
- **エラーハンドリング**: `HTTPException` を適切に使用し、クライアントに分かりやすいエラーメッセージを返す。

## 6. 開発ワークフロー
- **セットアップ**: `uv sync`
- **テスト実行**: `PYTHONPATH=. uv run pytest`
- **負荷テスト**: `./run_load_tests.sh`
- **UI起動**: `uv run streamlit run streamlit_app.py`
- **Docker起動**: `docker-compose up --build`

## 7. 注意事項
- **サブモジュールの依存**: `src/core/engine.py` にて `sys.path` を操作し、サブモジュールの `src` を追加している。この構造を維持すること。
- **モデルパス**: モデルは `extern/ndlocr-lite/src/model/` 以下のONNXファイルをデフォルトで参照する。
- **画像形式**: 入力は PIL Image を経由し、RGB形式でエンジンに渡される。
