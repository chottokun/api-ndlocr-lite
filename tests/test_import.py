import sys
from pathlib import Path

# Add submodule src to path
submodule_src = Path(__file__).parent.parent / "extern" / "ndlocr-lite" / "src"
sys.path.append(str(submodule_src))

def test_import_ndlocr():
    try:
        import ocr  # noqa: F401
        import deim  # noqa: F401
        import parseq  # noqa: F401
        print("Imports successful")
    except ImportError as e:
        print(f"Import failed: {e}")
        assert False
