from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_girf_and_vlmevalkit_have_separate_package_roots():
    assert (ROOT / "qwenvl" / "__init__.py").is_file()
    assert (ROOT / "VLMEvalKit" / "vlmeval" / "__init__.py").is_file()
    assert not (ROOT / "vlmeval").exists()


def test_evaluation_entry_points_and_helpers_exist():
    required = [
        "run.py",
        "VLMEvalKit/run.py",
        "scripts/eval/distribution_config.env",
        "scripts/eval/infer_backend.sh",
        "scripts/eval/run_dipe_eval.sh",
        "scripts/lib/cuda_env.sh",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, f"missing project files: {missing}"


def test_misnamed_evaluation_launcher_was_removed():
    assert not (ROOT / "scripts/eval/run_dipe_eval.py").exists()
