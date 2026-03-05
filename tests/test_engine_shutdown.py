import pytest
from unittest.mock import MagicMock, patch
from src.core.engine import NDLOCREngine

def test_engine_shutdown():
    # Mock _load_models to avoid loading real models
    with patch.object(NDLOCREngine, '_load_models'):
        engine = NDLOCREngine(device="cpu")

        # Mock the executor
        mock_executor = MagicMock()
        engine.executor = mock_executor

        # Call shutdown
        engine.shutdown()

        # Verify that executor.shutdown was called
        mock_executor.shutdown.assert_called_once()

def test_engine_shutdown_real_executor():
    # Another test with a real executor but still mocked _load_models
    with patch.object(NDLOCREngine, '_load_models'):
        engine = NDLOCREngine(device="cpu")

        executor = engine.executor
        assert not executor._shutdown

        engine.shutdown()

        assert executor._shutdown
