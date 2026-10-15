import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_vlmeval_package_is_discoverable_from_its_root():
    package = ROOT / "VLMEvalKit" / "vlmeval" / "__init__.py"
    spec = importlib.util.spec_from_file_location("bundled_vlmeval", package)
    assert spec is not None
    assert spec.loader is not None


def test_root_run_is_a_wrapper_not_a_second_vlmeval_copy():
    source = (ROOT / "run.py").read_text(encoding="utf-8")
    assert "VLMEvalKit" in source
    assert "runpy.run_path" in source
