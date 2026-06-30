import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "ocr_reader.py"
spec = importlib.util.spec_from_file_location("ocr_reader", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sanitize_text_for_speech_removes_problematic_unicode():
    text = "Hello—world! “This” is a test…"
    cleaned = module.sanitize_text_for_speech(text)
    assert "—" not in cleaned
    assert "“" not in cleaned
    assert "”" not in cleaned
    assert "…” not in cleaned
    assert cleaned == "Hello-world! This is a test..."
