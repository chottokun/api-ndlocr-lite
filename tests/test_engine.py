import pytest
from PIL import Image
from pathlib import Path
from src.core.engine import NDLOCREngine

@pytest.fixture(scope="module")
def engine():
    return NDLOCREngine(device="cpu")

def test_engine_ocr(engine):
    # Use a sample image from the submodule
    sample_img_path = Path("extern/ndlocr-lite/resource/digidepo_3048008_0025.jpg")
    if not sample_img_path.exists():
        pytest.skip(f"Sample image {sample_img_path} not found")
        
    img = Image.open(sample_img_path)
    result = engine.ocr(img, img_name="sample.jpg")
    
    assert "text" in result
    assert "lines" in result
    assert len(result["lines"]) > 0
    assert "第8章" in result["text"] or "職員" in result["text"]
    print(f"OCR Result Text: {result['text'][:100]}...")
