# **ndlocr-liteを基盤としたOpenAI互換OCR APIの実装計画：フォークおよびサブモジュール活用による開発ロードマップ**

国立国会図書館（NDL）が公開したndlocr-liteは、一般的なPC環境で動作可能な極めて軽量なOCRエンジンであり、これをAPI化することは、既存のLLMワークフローや文書管理システムへの統合において大きな価値を持ちます 1。本計画では、保守性とカスタマイズ性を両立させる「フォーク＋サブモジュール」方式を採用した具体的な実装手順を詳述します。

## **1\. リポジトリ構成と環境構築戦略**

開発の基盤として、エンジンのコアロジック（ndlocr-lite）と、インターフェース層（FastAPI）を分離して管理します。

### **ステップ1：リポジトリのフォークとサブモジュール化**

1. **フォーク（Fork）:** ndl-lab/ndlocr-liteを自身のリポジトリへフォークします。これにより、CLI前提のコードをAPIから呼び出しやすい形式（クラス化など）へ直接改修することが可能になります 1。  
2. **メインプロジェクトの作成:** FastAPIをベースとした新規リポジトリを作成します。  
3. **サブモジュールの追加:** メインプロジェクト内で以下のコマンドを実行し、フォークしたエンジンを特定のディレクトリ（例: extern/ndlocr-lite）に配置します。  
   * git submodule add \<自身のフォークURL\> extern/ndlocr-lite  
   * git submodule update \--init \--recursive

### **ステップ2：依存関係の統合管理**

ndlocr-liteのrequirements.txtに含まれる依存ライブラリ（onnxruntime, opencv-python, mmcvなど）を、メインプロジェクトのpyproject.tomlやrequirements.txtに統合します 1。特に、GPUを利用する場合はonnxruntime-gpuへの差し替えが必要です。

## **2\. コアロジックのラッピング（エンジンのクラス化）**

ndlocr-liteのsrc/ocr.pyは現状CLI実行を前提としたスクリプト構成となっています 1。これをAPIサーバーから効率的に呼び出すため、フォーク側のリポジトリで以下の改修を行います。

* **推論クラスの実装:** モデルのロード、画像の前処理、レイアウト認識、文字認識を一つのクラス（例: NDLOCRApp）としてカプセル化します。  
* **メモリ効率の最適化:** サーバー起動時に一度だけモデルをロードし、各リクエストで同じセッションを再利用する設計に変更します。  
* **出力形式の拡張:** 元のツールがサポートしているJSON、TXT、XML、TEIに加え、OpenAI/Mistral互換のレスポンスを生成するための生データ（座標付きテキスト）を直接取得できるメソッドを追加します。

## **3\. FastAPIによるAPIインターフェースの実装**

FastAPI層では、外部との通信とリクエストのバリデーションを担当します。

### **ライフサイクル管理（Lifespan）**

サーバーの起動・終了イベントに合わせて、ocr.pyからインポートした推論クラスのインスタンス化と解放を行います。これにより、リクエストごとのモデルロードによる遅延（数秒〜数十秒）を排除します。

### **エンドポイント設計（/v1/ocr）**

OpenAIやMistral AIのAPI仕様に準拠したエンドポイントを構築します。

* **入力処理:** multipart/form-dataによる画像/PDFの直接アップロードと、JSON内でのBase64文字列の両方をサポートします。  
* **非同期処理:** OCR処理は計算負荷が高いため、FastAPIのBackgroundTasksまたはCelery \+ Redisを用いた非同期タスクキュー方式を採用します。クライアントへは即座にjob\_idを返し、後ほど /v1/ocr/jobs/{job\_id} で結果を取得するフローを構築します。

## **4\. スキーマ定義とOpenAI互換性の確保**

レスポンスのJSON構造を業界標準に合わせることで、既存のクライアントライブラリからの利用を容易にします。

* **ページ単位の構造化:** pages配列の中に、各ページのindex、markdown（テキスト内容）、およびレイアウト情報を含めます。  
* **座標情報の付与:** ndlocr-liteが抽出するバウンディングボックス情報を、以下の数学的表現に基づき返却します。  
  ![][image1]  
* **メタデータの提供:** 処理に使用したモデル名、処理ページ数、消費トークン（推定値）などのusage\_infoを付与します。

## **5\. パフォーマンスとデプロイメントの最適化**

### **ハードウェアアクセラレーション**

ndlocr-liteはCPUでも高速に動作しますが、大規模なバッチ処理を行う場合はGPU（CUDA/TensorRT）プロバイダーを有効にする設定を環境変数で切り替えられるようにします。

### **コンテナ化（Docker）**

Python 3.10環境をベースに、onnxruntimeのランタイムを含む軽量なDockerイメージを構築します。モデルファイル（約6.7GB）はイメージ内に含めるか、起動時にボリュームマウントする構成をとります。

| 開発フェーズ | 主な作業内容 | 期待される成果 |
| :---- | :---- | :---- |
| **Phase 1: 基盤** | リポジトリのフォークとサブモジュール設定、依存関係の解決 | エンジンとAPIの分離管理体制の確立 |
| **Phase 2: 統合** | ocr.pyのクラス化、FastAPIからのインポートとライフサイクル実装 | モデル常駐による高速な推論準備 |
| **Phase 3: 互換** | Pydanticを用いたOpenAI/Mistral互換スキーマの実装 | 標準的なSDKからの接続可能性 |
| **Phase 4: 運用** | タスクキューによる非同期処理、Docker化、スケーリング設定 | 商用レベルの安定稼働と拡張性 |

## **結論**

本計画に基づき、ndlocr-liteをフォークしてサブモジュールとして統合することで、本家の更新を適宜取り込みつつ、APIとして最適化された独自の改修を施すことが可能になります。これは、プライバシーが重視される環境や、特定ドメインの日本語資料を大量に扱うAIエージェントの基盤として、極めて堅牢なソリューションとなります。

#### **引用文献**

1. NDLOCR‑Lite application repository (including source code) \- GitHub, 2月 24, 2026にアクセス、 [https://github.com/ndl-lab/ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite)  
2. GPUなしで動作する軽量なAI OCRツール「NDLOCR-Lite」、国会図書館のラボから無償公開, 2月 24, 2026にアクセス、 [https://forest.watch.impress.co.jp/docs/news/2088188.html](https://forest.watch.impress.co.jp/docs/news/2088188.html)  
3. mistral-ocr-latest \- AI/ML API Documentation, 2月 24, 2026にアクセス、 [https://docs.aimlapi.com/api-references/vision-models/ocr-optical-character-recognition/mistral-ai/mistral-ocr-latest](https://docs.aimlapi.com/api-references/vision-models/ocr-optical-character-recognition/mistral-ai/mistral-ocr-latest)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAj4AAAA4CAYAAAD99PKYAAALm0lEQVR4Xu2dCayt1xTHl5jneYqpldaLWaOGmqlqRRFDlBjyUmqs8SlBxUUlbYPioVJTkZopMZaGWySKREJQ0YpXEYKIREpitn9vf6tn332+c9697usd3vn9kpV7zjfuvb/vnPX/1lr73AgRERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERFZHK5c7GbFblnsBt06ka1Oe/9ep1snIiIyBQ7je8U+WOyZ3TqRrQ5i/dRi5xZ7abdORGRhuF+xXxf7b2OXFfvt8Prfxb5U7A65w8CNip0XdX3ux+vfFPvH8P7nxZ5Q7ErDPtsdhM+Hwqdl2d7cs9hz+4UiIovGO4r9rdi9uuWHFvtlsYuK3bpbBw8v9p9ir+6WX7PY2VHF0GNWrtq2KHzkQEDhIyILD458udhPi91k5aq94OyJ4Dy6X1F4ZVThgwDqYXv2Y/8DAYWPHAgofERk4bl9sd9FrV3p01Ipiv5e7IiVq+IqxT5dbE+xW61ctRdEEcJnqVu+XVH4yIGAwkdEFh5SUQiU5/QrCsdFTVe9M6rQaUHs7IlaA3SNlavioKgpsh/FeIrsqsUOL/bEqPVDveBi/YMG4zUzUvjCfsDwejOYJXyybY8odu2ofaFPR0VN+W01cmxpczuWtPuG3bKtwm2KHTv8hRsXe2TUa7KVob20k/a28Hnp76ONQuEjIgvPm6JGdNKRYLcr9rpivy/2lBh3hoiQfxY7LSb7YY8v9qtiu4td7/KtKzhXxNQlxV5W7MnFvlrszKgOGTg3M0+eHbX26IJi7y12UtSC6rcM2200Y8Ln+sU+GrUvr4kq9N4WdUzfH7UfvSjcTGjvOcVeXuxnsXJ22kOK/TmmI3ubCffLS6LeH8+I2ubTi3242CuKXVxsx+Vbby34HHC/8Pngvkjxw/3AffGJGP9cXdEofERkoclU1h+jprrOGgynTc3PKTEtXpKs7+FLPPfDvl3sc8XuNNl0LzixXVHPde9mOQXVONxjokZI3lPskGEdX9J/jZouy0LqsZRcwmwzhBQz1VZrr927577phQ9tYHwe2Ky/tNjHit2t2J+iju1mPdmPQVTv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>