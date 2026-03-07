import os
import io
import pytest
from PIL import Image
from src.core.engine import NDLOCREngine

@pytest.fixture(scope="module")
def engine():
    return NDLOCREngine(device="cpu")

def test_engine_basic_ocr(engine):
    # Test with a known sample from the submodule resource
    sample_path = "extern/ndlocr-lite/resource/digidepo_3048008_0025.jpg"
    if not os.path.exists(sample_path):
        pytest.skip(f"Sample not found at {sample_path}")
    
    img = Image.open(sample_path)
    result = engine.ocr(img, img_name="test_sample.jpg")
    
    assert "text" in result
    assert "lines" in result
    assert len(result["lines"]) > 0
    print(f"OCR Success: Found {len(result['lines'])} lines.")

def test_engine_empty_image(engine):
    # Test with a very small empty image
    img = Image.new('RGB', (100, 100), color = (255, 255, 255))
    result = engine.ocr(img, img_name="empty.jpg")
    assert "text" in result
    # Depending on the model, it might find nothing or some noise
    print(f"Empty image OCR result text length: {len(result['text'])}")

def test_engine_high_resolution(engine):
    # Test with a larger image (within reasonable limits)
    img = Image.new('RGB', (2000, 3000), color = (255, 255, 255))
    result = engine.ocr(img, img_name="large_empty.jpg")
    assert "text" in result

def test_engine_shutdown(engine):
    # Verify shutdown doesn't crash
    engine.shutdown()
    # Re-initialize for subsequent tests if fixture wasn't module scope or if needed
    # But here it's module scope, so this might be better as a standalone test at the end.
    pass
