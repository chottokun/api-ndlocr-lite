import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
from src.core.engine import NDLOCREngine

@pytest.fixture
def engine():
    # Mocking models to avoid loading them and using CPU/GPU
    with patch('src.core.engine.DEIM'), \
         patch('src.core.engine.PARSEQ'), \
         patch('src.core.engine.safe_load'):
        engine = NDLOCREngine(device="cpu")
        engine.detector = MagicMock()
        engine.recognizer100 = MagicMock()
        engine.recognizer30 = MagicMock()
        engine.recognizer50 = MagicMock()
        # Mocking the detector's classes as it's used in ocr
        engine.detector.classes = {0: "text_block", 1: "line_main"}
        return engine

def test_ocr_fallback_behavior(engine):
    # Mock image
    img = Image.new('RGB', (100, 100))

    # Mock detector results
    engine.detector.detect.return_value = [
        {"box": [0, 0, 50, 10], "confidence": 0.9, "class_index": 1}
    ]

    # Custom XML string with missing/invalid attributes
    # We want to test LINE elements.
    # Note: convert_to_xml_string3 usually produces the PAGE and its children.
    # The ocr method wraps it in <OCRDATASET>.

    # Case 1: PRED_CHAR_CNT is missing, CONF is missing
    xml_with_missing_attrs = """
    <PAGE IMAGENAME="test.jpg" WIDTH="100" HEIGHT="100">
        <LINE X="0" Y="0" WIDTH="50" HEIGHT="10" TYPE="本文">
        </LINE>
    </PAGE>
    """

    # Case 2: PRED_CHAR_CNT is invalid, CONF is invalid
    xml_with_invalid_attrs = """
    <PAGE IMAGENAME="test.jpg" WIDTH="100" HEIGHT="100">
        <LINE X="0" Y="10" WIDTH="50" HEIGHT="10" TYPE="本文" PRED_CHAR_CNT="invalid" CONF="none">
        </LINE>
    </PAGE>
    """

    def mock_convert(img_w, img_h, img_name, classeslist, resultobj):
        return xml_with_missing_attrs + xml_with_invalid_attrs

    # Mock recognizers to return something
    engine.recognizer100.read.return_value = "recognized text"

    with patch('src.core.engine.convert_to_xml_string3', side_effect=mock_convert), \
         patch('src.core.engine.eval_xml'): # skip eval_xml as it might fail on our custom xml

        result = engine.ocr(img, img_name="test.jpg")

    assert len(result["lines"]) == 2

    # Check Case 1 (missing attributes)
    # PRED_CHAR_CNT fallback to 100.0 is internal but we can verify it didn't crash
    # and it should have used recognizer100 because 100.0 is not 2 or 3 (cascade triggers)
    assert result["lines"][0]["confidence"] == 0.0

    # Check Case 2 (invalid attributes)
    assert result["lines"][1]["confidence"] == 0.0

    # Verify that it didn't crash and processed everything
    assert result["lines"][0]["text"] == "recognized text"
    assert result["lines"][1]["text"] == "recognized text"

def test_ocr_fallback_in_cascade(engine):
    # Test if pred_char_cnt fallback actually works and routes to correct recognizer
    img = Image.new('RGB', (100, 100))
    engine.detector.detect.return_value = [{"box": [0, 0, 50, 10], "confidence": 0.9, "class_index": 1}]

    xml = """
    <PAGE IMAGENAME="test.jpg" WIDTH="100" HEIGHT="100">
        <LINE X="0" Y="0" WIDTH="50" HEIGHT="10" TYPE="本文" PRED_CHAR_CNT="invalid">
        </LINE>
    </PAGE>
    """

    engine.recognizer100.read.return_value = "rec100"
    engine.recognizer30.read.return_value = "rec30"
    engine.recognizer50.read.return_value = "rec50"

    with patch('src.core.engine.convert_to_xml_string3', return_value=xml), \
         patch('src.core.engine.eval_xml'):

        result = engine.ocr(img, img_name="test.jpg")

    # Since PRED_CHAR_CNT was invalid, it should fall back to 100.0
    # In _process_cascade, 100.0 goes to targetdflist100.
    # So recognizer100.read should have been called.
    engine.recognizer100.read.assert_called()
    assert result["lines"][0]["text"] == "rec100"
