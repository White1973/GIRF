#!/usr/bin/env python3
"""Filter explicitly gated datasets that the current HF account cannot download."""

import os
import sys
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError


REPOSITORIES = {
    "LongVideoBench_8frame": "longvideobench/LongVideoBench",
    "MLVU_8frame": "MLVU/MVLU",
}


def load_local_env() -> None:
    env_path = Path(
        os.environ.get(
            "VLMEVALKIT_DIR",
            Path(__file__).resolve().parents[2] / "VLMEvalKit",
        )
    ) / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def is_denied(error: BaseException) -> bool:
    if isinstance(error, GatedRepoError):
        return True
    return isinstance(error, HfHubHTTPError) and getattr(error.response, "status_code", None) in {401, 403}


def main() -> int:
    load_local_env()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    for dataset in sys.argv[1:]:
        repository = REPOSITORIES.get(dataset)
        if repository is None:
            print(dataset)
            continue
        try:
            url = hf_hub_url(repository, ".gitattributes", repo_type="dataset")
            get_hf_file_metadata(url, token=token)
        except Exception as error:
            if is_denied(error):
                print(
                    f"SKIP {dataset}: Hugging Face returned gated-access "
                    f"{getattr(getattr(error, 'response', None), 'status_code', 403)}.",
                    file=sys.stderr,
                )
                continue
            print(
                f"WARN {dataset}: access preflight failed ({type(error).__name__}); "
                "keeping it in the evaluation list.",
                file=sys.stderr,
            )
        print(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
