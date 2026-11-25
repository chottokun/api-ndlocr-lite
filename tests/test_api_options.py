import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.core.engine import NDLOCREngine


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """テスト用の最小 JPEG 画像バイト列を生成する。"""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buf, format="JPEG")
    return buf.getvalue()


def _make_mock_engine_with_response():
    """呼び出し引数に応じて結果を返すモックエンジンを生成する。"""
    mock_engine = MagicMock()

    def fake_ocr(
        img,
        img_name="image.jpg",
        enable_character_inclusion=None,
        enable_rich_markdown=None,
        enable_table_structure=None,
    ):
        text = "旧字体テスト"
        if enable_character_inclusion:
            text = "新字体テスト"
        markdown = f"# {text}" if enable_rich_markdown else text
        res = {
            "text": text,
            "markdown": markdown,
            "lines": [
                {
                    "id": 0,
                    "text": text,
                    "confidence": 0.99,
                    "boundingBox": [[0, 0], [0, 10], [50, 10], [50, 0]],
                    "isVertical": "false",
                    "isTextline": "true",
                    "class_index": 1,
                }
            ],
            "img_info": {"width": img.width, "height": img.height, "name": img_name},
        }
        if enable_table_structure:
            res["tables"] = [
                {
                    "table_index": 1,
                    "layout_bbox": [0, 0, 50, 50],
                    "html": "<table><tr><td>A</td></tr></table>",
                    "markdown": "| A |\n| --- |",
                    "num_cells": 1,
                }
            ]
        return res

    mock_engine.ocr.side_effect = fake_ocr
    return mock_engine


# ---------------------------------------------------------------------------
# 1. .env 読み込みと環境変数優先順位のテスト
# ---------------------------------------------------------------------------


def test_dotenv_loaded_as_default_and_env_takes_precedence(tmp_path, monkeypatch):
    """一時 .env ファイルがデフォルトとして読み込まれ、既存の環境変数が優先されることを検証する。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ENABLE_CHARACTER_INCLUSION=true\n"
        "ENABLE_RICH_MARKDOWN=true\n"
        "ENABLE_TABLE_STRUCTURE=false\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENABLE_CHARACTER_INCLUSION", raising=False)
    monkeypatch.delenv("ENABLE_RICH_MARKDOWN", raising=False)
    # 既存の環境変数で ENABLE_TABLE_STRUCTURE=true を設定（.env の false より優先されるべき）
    monkeypatch.setenv("ENABLE_TABLE_STRUCTURE", "true")

    with patch("src.api.main.NDLOCREngine") as mock_engine_cls:
        mock_engine_cls.return_value = MagicMock()
        with TestClient(app):
            pass

        mock_engine_cls.assert_called_once_with(
            device="cpu",
            enable_tcy=False,
            enable_character_inclusion=True,
            enable_rich_markdown=True,
            enable_table_structure=True,
        )


# ---------------------------------------------------------------------------
# 2. /v1/ocr (multipart + クエリパラメータ) のテスト
# ---------------------------------------------------------------------------


def test_ocr_multipart_default_and_query_overrides():
    """未指定時は None（位置引数のみ）が渡され、クエリ指定時は各フラグが上書きされることを検証する。"""
    img_bytes = _make_jpeg_bytes()
    mock_engine = _make_mock_engine_with_response()

    with patch("src.api.main.NDLOCREngine", return_value=mock_engine):
        with TestClient(app) as client:
            # 1) 未指定：従来通り位置引数のみで呼ばれること（後方互換性）
            res_default = client.post(
                "/v1/ocr",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
            assert res_default.status_code == 200
            assert res_default.json()["pages"][0]["markdown"] == "旧字体テスト"
            assert res_default.json()["pages"][0]["tables"] is None
            last_call = mock_engine.ocr.call_args
            assert last_call.kwargs == {}

            # 2) クエリパラメータで全有効化
            res_on = client.post(
                "/v1/ocr?character_inclusion=true&rich_markdown=true&table_structure=true",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
            assert res_on.status_code == 200
            data_on = res_on.json()
            assert data_on["pages"][0]["markdown"] == "# 新字体テスト"
            assert len(data_on["pages"][0]["tables"]) == 1

            # 3) クエリパラメータで明示的に false を指定（デフォルト true 想定の上書き）
            res_off = client.post(
                "/v1/ocr?character_inclusion=false&rich_markdown=false&table_structure=false",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
            assert res_off.status_code == 200
            last_call_off = mock_engine.ocr.call_args
            assert last_call_off.kwargs == {
                "enable_character_inclusion": False,
                "enable_rich_markdown": False,
                "enable_table_structure": False,
            }

            # 4) 不正なブール値クエリは 422
            res_invalid = client.post(
                "/v1/ocr?rich_markdown=notabool",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
            assert res_invalid.status_code == 422


# ---------------------------------------------------------------------------
# 3. /v1/ocr (JSON Base64 + JSON ボディ / クエリ優先順位) のテスト
# ---------------------------------------------------------------------------


def test_ocr_json_body_options_and_query_precedence():
    """JSON ボディ指定とクエリ指定が競合した場合にクエリが優先され、不正型が 400 になることを検証する。"""
    encoded = base64.b64encode(_make_jpeg_bytes()).decode()
    mock_engine = _make_mock_engine_with_response()

    with patch("src.api.main.NDLOCREngine", return_value=mock_engine):
        with TestClient(app) as client:
            # JSON ボディのみで有効化
            res_body = client.post(
                "/v1/ocr",
                json={
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "character_inclusion": True,
                    "rich_markdown": True,
                    "table_structure": True,
                },
            )
            assert res_body.status_code == 200
            assert res_body.json()["pages"][0]["markdown"] == "# 新字体テスト"
            assert len(res_body.json()["pages"][0]["tables"]) == 1

            # クエリ（false）と JSON ボディ（true）が競合 → クエリが勝つ
            res_override = client.post(
                "/v1/ocr?rich_markdown=false&table_structure=false",
                json={
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "character_inclusion": True,
                    "rich_markdown": True,
                    "table_structure": True,
                },
            )
            assert res_override.status_code == 200
            assert res_override.json()["pages"][0]["markdown"] == "新字体テスト"
            assert res_override.json()["pages"][0]["tables"] is None

            # 不正型は 400
            res_bad_type = client.post(
                "/v1/ocr",
                json={
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "rich_markdown": {"bad": 1},
                },
            )
            assert res_bad_type.status_code == 400


# ---------------------------------------------------------------------------
# 4. /v1/ocr/jobs (非同期ジョブ) のテスト
# ---------------------------------------------------------------------------


def test_ocr_jobs_propagate_query_and_body_options():
    """非同期ジョブ作成時のクエリおよび JSON ボディオプションが結果に反映されることを検証する。"""
    img_bytes = _make_jpeg_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    mock_engine = _make_mock_engine_with_response()

    with patch("src.api.main.NDLOCREngine", return_value=mock_engine):
        with TestClient(app) as client:
            # クエリパラメータ経由
            res_q = client.post(
                "/v1/ocr/jobs?character_inclusion=true&rich_markdown=true&table_structure=true",
                files={"file": ("job.jpg", img_bytes, "image/jpeg")},
            )
            assert res_q.status_code == 200
            job_id_q = res_q.json()["job_id"]

            time.sleep(0.1)
            status_q = client.get(f"/v1/ocr/jobs/{job_id_q}").json()
            assert status_q["status"] == "completed"
            assert status_q["result"]["pages"][0]["markdown"] == "# 新字体テスト"
            assert len(status_q["result"]["pages"][0]["tables"]) == 1

            # JSON ボディ＋クエリ上書き
            res_b = client.post(
                "/v1/ocr/jobs?rich_markdown=false",
                json={
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "character_inclusion": True,
                    "rich_markdown": True,
                    "table_structure": False,
                },
            )
            assert res_b.status_code == 200
            job_id_b = res_b.json()["job_id"]

            time.sleep(0.1)
            status_b = client.get(f"/v1/ocr/jobs/{job_id_b}").json()
            assert status_b["status"] == "completed"
            assert status_b["result"]["pages"][0]["markdown"] == "新字体テスト"
            assert status_b["result"]["pages"][0]["tables"] is None


# ---------------------------------------------------------------------------
# 5. /v1/chat/completions (OpenAI 互換 API) のテスト
# ---------------------------------------------------------------------------


def test_openai_chat_completions_options_and_precedence():
    """OpenAI 互換エンドポイントで後方互換・JSON 指定・クエリ優先順位が機能することを検証する。"""
    encoded = base64.b64encode(_make_jpeg_bytes()).decode()
    mock_engine = _make_mock_engine_with_response()
    base_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "OCR please"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }
    ]

    with patch("src.api.main.NDLOCREngine", return_value=mock_engine):
        with TestClient(app) as client:
            # 未指定（従来通り）
            res_default = client.post(
                "/v1/chat/completions",
                json={"model": "ndlocr-lite", "messages": base_messages},
            )
            assert res_default.status_code == 200
            assert res_default.json()["choices"][0]["message"]["content"] == "旧字体テスト"
            assert mock_engine.ocr.call_args.kwargs == {}

            # JSON extra_body 相当の指定
            res_body = client.post(
                "/v1/chat/completions",
                json={
                    "model": "ndlocr-lite",
                    "messages": base_messages,
                    "character_inclusion": True,
                    "rich_markdown": True,
                    "table_structure": True,
                },
            )
            assert res_body.status_code == 200
            assert res_body.json()["choices"][0]["message"]["content"] == "# 新字体テスト"

            # クエリパラメータで上書き
            res_query = client.post(
                "/v1/chat/completions?rich_markdown=false",
                json={
                    "model": "ndlocr-lite",
                    "messages": base_messages,
                    "character_inclusion": True,
                    "rich_markdown": True,
                },
            )
            assert res_query.status_code == 200
            assert res_query.json()["choices"][0]["message"]["content"] == "新字体テスト"


# ---------------------------------------------------------------------------
# 6. 表構造化モデルの遅延ロード＆スレッドセーフ性テスト
# ---------------------------------------------------------------------------


def test_table_recognizer_lazy_loading_and_thread_safety():
    """enable_table_structure=False で起動後、必要時に 1 回だけスレッドセーフに遅延ロードされることを検証する。"""
    with patch("src.core.engine.DEIM"), \
         patch("src.core.engine.PARSEQ"), \
         patch("src.core.engine.safe_load"), \
         patch("src.core.engine.TableCenterNetRecognizer") as mock_table_cls:
        mock_table_cls.return_value = MagicMock()
        engine = NDLOCREngine(device="cpu", enable_table_structure=False)

        # 初期状態ではロードされていないこと
        assert engine.table_recognizer is None
        mock_table_cls.assert_not_called()

        # 8 スレッドから同時に _ensure_table_recognizer を呼び出しても 1 回だけ初期化されること
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: engine._ensure_table_recognizer(), range(16)))

        mock_table_cls.assert_called_once()
        assert all(r is results[0] for r in results)
        assert engine.table_recognizer is not None


# ---------------------------------------------------------------------------
# 7. 実モデル＋サンプル画像による E2E 結合テスト
# ---------------------------------------------------------------------------


def test_e2e_sample_image_options_and_isolation(tmp_path, monkeypatch):
    """実モデルとサンプル画像でデフォルト OFF -> リクエスト時全 ON -> デフォルト OFF の状態分離を検証する。"""
    sample_path = Path("extern/ndlocr-lite/resource/digidepo_3048008_0025.jpg").resolve()
    if not sample_path.exists():
        pytest.skip(f"Sample image {sample_path} not found")

    # ローカルの .env の影響を受けないよう空ディレクトリへ移動し、環境変数も明示的に false に揃える
    monkeypatch.chdir(tmp_path)
    for key in (
        "ENABLE_TCY",
        "ENABLE_CHARACTER_INCLUSION",
        "ENABLE_RICH_MARKDOWN",
        "ENABLE_TABLE_STRUCTURE",
    ):
        monkeypatch.setenv(key, "false")

    img_bytes = sample_path.read_bytes()

    with TestClient(app) as client:
        # 初期状態では表モデルはロードされていない
        assert app.state.engine.table_recognizer is None

        # 1) デフォルト (OFF)
        res_off = client.post(
            "/v1/ocr",
            files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
        )
        assert res_off.status_code == 200
        page_off = res_off.json()["pages"][0]
        assert page_off["tables"] is None
        assert "專門" in page_off["markdown"] or "一覽" in page_off["markdown"]

        # 2) リクエスト単位で全 ON（表モデルが遅延ロードされる）
        res_on = client.post(
            "/v1/ocr?character_inclusion=true&rich_markdown=true&table_structure=true",
            files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
        )
        assert res_on.status_code == 200
        page_on = res_on.json()["pages"][0]
        assert app.state.engine.table_recognizer is not None
        assert page_on["tables"] is not None
        assert len(page_on["tables"]) >= 1
        assert "<table" in page_on["tables"][0]["html"]
        assert "|" in page_on["tables"][0]["markdown"]
        # 異体字包摂が適用されていること（專門→専門、一覽→一覧 等）
        assert "專門" not in page_on["markdown"]
        assert "一覽" not in page_on["markdown"]
        assert "専門" in page_on["markdown"] or "一覧" in page_on["markdown"]

        # 3) 直後に再び未指定で呼び出してもデフォルト (OFF) に戻ること（状態汚染なし）
        res_after = client.post(
            "/v1/ocr",
            files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
        )
        assert res_after.status_code == 200
        page_after = res_after.json()["pages"][0]
        assert page_after["tables"] is None
        assert page_after["markdown"] == page_off["markdown"]
