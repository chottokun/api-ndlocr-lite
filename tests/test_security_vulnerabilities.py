import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.core.engine import NDLOCREngine


def _create_mock_engine(**kwargs) -> NDLOCREngine:
    """モデルロードをモックして NDLOCREngine インスタンスを生成するヘルパー。"""
    with patch("src.core.engine.DEIM"), \
         patch("src.core.engine.PARSEQ"), \
         patch("src.core.engine.safe_load"):
        engine = NDLOCREngine(device="cpu", **kwargs)
        engine.detector = MagicMock()
        engine.detector.classes = {0: "text_block", 1: "line_main"}
        engine.detector.markdown_styles = {0: "other", 1: "join_next", 4: "h1"}
        engine.recognizer100 = MagicMock()
        engine.recognizer30 = MagicMock()
        engine.recognizer50 = MagicMock()
        return engine


def test_img_name_xml_injection_sanitized():
    """img_name に XML インジェクション文字列が含まれてもサニタイズされることを検証する。"""
    img = Image.new("RGB", (100, 100))
    engine = _create_mock_engine()
    engine.detector.detect.return_value = []

    captured = {}

    def fake_convert(img_w, img_h, img_name, classeslist, resultobj):
        captured["img_name"] = img_name
        return f'<PAGE IMAGENAME="{img_name}" WIDTH="{img_w}" HEIGHT="{img_h}"></PAGE>'

    with patch("src.core.engine.convert_to_xml_string3", side_effect=fake_convert), \
         patch("src.core.engine.eval_xml"):
        result = engine.ocr(img, img_name='evil" /><LINE TYPE="injected" />')

    # サニタイズ後の img_name には XML 特殊文字（<, >, ", /, =）が含まれないこと
    sanitized = captured["img_name"]
    for dangerous_char in ('<', '>', '"', '/', '='):
        assert dangerous_char not in sanitized
    assert result["lines"] == []


def test_path_traversal_filename_in_upload():
    """アップロードファイル名にパストラバーサル文字が含まれても安全に処理されることを検証する。"""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buf, format="JPEG")
    buf.seek(0)

    mock_engine = MagicMock()
    mock_engine.ocr.return_value = {
        "text": "テスト",
        "markdown": "テスト",
        "lines": [],
        "img_info": {"width": 64, "height": 64, "name": "../../../etc/passwd.jpg"},
    }

    with patch("src.api.main.NDLOCREngine", return_value=mock_engine):
        with TestClient(app) as client:
            response = client.post(
                "/v1/ocr",
                files={"file": ("../../../etc/passwd.jpg", buf, "image/jpeg")},
            )

    assert response.status_code == 200
    assert response.json()["pages"][0]["markdown"] == "テスト"


def test_pred_markdown_malicious_and_out_of_range_values():
    """PRED_MARKDOWN に不正値や範囲外数値が入ってもクラッシュせずフォールバックすることを検証する。"""
    img = Image.new("RGB", (200, 200))
    xml = """
    <PAGE IMAGENAME="test.jpg" WIDTH="200" HEIGHT="200">
        <TEXTBLOCK CONF="0.9">
            <LINE X="10" Y="10" WIDTH="100" HEIGHT="20" TYPE="本文" CONF="0.9" PRED_CHAR_CNT="10" PRED_MARKDOWN="invalid" ORDER="1" />
            <LINE X="10" Y="40" WIDTH="100" HEIGHT="20" TYPE="本文" CONF="0.9" PRED_CHAR_CNT="10" PRED_MARKDOWN="-1" ORDER="2" />
            <LINE X="10" Y="70" WIDTH="100" HEIGHT="20" TYPE="本文" CONF="0.9" PRED_CHAR_CNT="10" PRED_MARKDOWN="999999" ORDER="3" />
        </TEXTBLOCK>
    </PAGE>
    """

    engine = _create_mock_engine(enable_rich_markdown=True)
    engine.detector.detect.return_value = []
    engine.recognizer100.read.side_effect = ["行1", "行2", "行3"]

    with patch("src.core.engine.convert_to_xml_string3", return_value=xml), \
         patch("src.core.engine.eval_xml"):
        result = engine.ocr(img, img_name="test.jpg")

    assert "行1" in result["markdown"]
    assert "行2" in result["markdown"]
    assert "行3" in result["markdown"]


def test_table_html_and_markdown_escape_xss():
    """表の HTML / Markdown 変換で XSS ペイロードやパイプ記号がエスケープされることを検証する。"""
    from tablecenternet import cells_to_html, cells_to_markdown

    cells = [
        {
            "logic_axis_0based": [0, 0, 0, 0],
            "score": 0.99,
            "corners": [[0, 0], [100, 0], [100, 50], [0, 50]],
        }
    ]
    ocr_lines = [
        {
            "text": '<script>alert("xss")</script>|col',
            "boundingBox": [[10, 10], [10, 20], [50, 20], [50, 10]],
        }
    ]

    html_out = cells_to_html(cells, ocr_lines)
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out

    md_out = cells_to_markdown(cells, ocr_lines)
    assert "\\|col" in md_out


def test_internal_exception_does_not_leak_details():
    """OCR エンジン内部で例外が発生した際、内部パス等の詳細がクライアントに漏洩しないことを検証する。"""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buf, format="JPEG")
    buf.seek(0)

    mock_engine = MagicMock()
    mock_engine.ocr.side_effect = RuntimeError("SECRET_INTERNAL_PATH=/etc/shadow")

    with patch("src.api.main.NDLOCREngine", return_value=mock_engine):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/ocr",
                files={"file": ("test.jpg", buf, "image/jpeg")},
            )

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "An internal error occurred during OCR processing"
    assert "SECRET_INTERNAL_PATH" not in response.text
    assert "/etc/shadow" not in response.text
