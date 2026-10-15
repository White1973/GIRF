#!/usr/bin/env bash
# Recreate the Python dependencies used by the project's existing `eval` Conda environment.
# Usage: bash scripts/install_eval_env.sh [target_environment_name]
# Default target environment: eval
#
# Requirements: Conda/Mamba, Linux x86_64, Python 3.12, and a CUDA 12.8-capable
# NVIDIA driver when GPU inference is required.  VLMEvalKit is installed from
# this repository at the same revision-independent local path.

set -euo pipefail

ENV_NAME="${1:-eval}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required but was not found in PATH." >&2
    exit 1
fi

if [[ -d "${PROJECT_ROOT}/VLMEvalKit" ]]; then
    VLM_EVAL_PATH="${PROJECT_ROOT}/VLMEvalKit"
else
    echo "Missing ${PROJECT_ROOT}/VLMEvalKit; it is required for the editable vlmeval install." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Updating existing Conda environment: ${ENV_NAME}"
else
    echo "Creating Conda environment: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" python=3.12 pip setuptools wheel
fi

# PyTorch's CUDA-12.8 wheels are hosted on its own index. Install this first so
# packages such as vLLM and xFormers resolve against the intended ABI. PyTorch's
# download endpoint currently redirects wheel requests to download-r2.pytorch.org;
# trust both hosts so TLS-inspecting/self-signed proxies do not break the
# redirected request. Reusing an already-correct environment must not reinstall
# or silently replace this stack.
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
if conda run -n "${ENV_NAME}" python -c \
    'import torch, torchvision; raise SystemExit(0 if (torch.__version__.split("+")[0] == "2.8.0" and torchvision.__version__.split("+")[0] == "0.23.0" and torch.version.cuda == "12.8") else 1)' \
    >/dev/null 2>&1; then
    echo "Keeping existing PyTorch 2.8.0 / TorchVision 0.23.0 CUDA 12.8 stack."
else
    conda run -n "${ENV_NAME}" python -m pip install \
          --index-url https://download.pytorch.org/whl/cu128 \
          --extra-index-url https://pypi.org/simple \
          --trusted-host download.pytorch.org \
          --trusted-host download-r2.pytorch.org \
          "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"
fi

# hf-gradio 0.4.x requires gradio-client>=2, while the working Gradio 5.50.0
# stack requires gradio-client==1.14.0. It is not used by this evaluation
# pipeline, but may be left behind by an earlier failed installation.
conda run -n "${ENV_NAME}" python -m pip uninstall -y hf-gradio >/dev/null 2>&1 || true

# Install the resolver-sensitive web stack as one tested compatibility unit.
# These are the exact versions in the currently working `eval` environment.
conda run -n "${ENV_NAME}" python -m pip install --upgrade-strategy only-if-needed \
      "fastapi==0.115.12" \
      "starlette==0.46.2" \
      "prometheus-fastapi-instrumentator==7.1.0" \
      "pydantic==2.12.3" \
      "pydantic_core==2.41.4" \
      "google-genai==1.56.0" \
      "gradio==5.50.0" \
      "gradio_client==1.14.0"

# Exact package versions exported from `conda run -n eval python -m pip freeze --all`
# on 2026-08-17.  `pip` itself and the local editable VLMEvalKit package are
# handled separately above/below.
PACKAGES=(
  absl-py==2.5.0
  accelerate==1.14.0
  aiofiles==24.1.0
  aiohappyeyeballs==2.7.1
  aiohttp==3.14.3
  aiosignal==1.4.0
  anls==0.0.2
  annotated-doc==0.0.5
  annotated-types==0.8.0
  antlr4-python3-runtime==4.11.1
  anyio==4.14.2
  apted==1.0.3
  astor==0.8.1
  asttokens==3.0.2
  attrs==26.1.0
  av==18.0.0
  bert-score==0.3.13
  blake3==1.0.9
  brotli==1.2.0
  cachetools==7.1.7
  cairocffi==1.7.1
  CairoSVG==2.9.0
  cbor2==6.1.4
  cd-fvd==0.1.1
  certifi==2026.7.22
  cffi==2.1.1
  charset-normalizer==3.4.9
  click==8.4.2
  cloudpickle==3.1.2
  colormath==3.0.0
  compressed-tensors==0.11.0
  contourpy==1.3.3
  cryptography==50.0.0
  cssselect2==0.9.0
  cuda-pathfinder==1.6.0
  cupy-cuda12x==14.1.1
  cycler==0.12.1
  datasets==5.0.1
  decorator==5.3.1
  decord==0.6.0
  defusedxml==0.7.1
  depyf==0.19.0
  detect-installer==0.1.0
  dill==0.4.1
  diskcache==5.6.3
  Distance==0.1.3
  distro==1.9.0
  dnspython==2.8.0
  docopt==0.6.2
  dotenv==0.9.9
  editdistance==0.8.1
  einops==0.8.2
  email-validator==2.3.0
  et_xmlfile==2.0.0
  executing==2.2.1
  fastapi-cli==0.0.32
  fastapi-cloud-cli==0.23.0
  fastar==0.11.0
  fastuuid==0.14.0
  ffmpy==1.0.0
  filelock==3.32.2
  fonttools==4.63.0
  frozendict==2.4.7
  frozenlist==1.8.0
  fsspec==2026.6.0
  ftfy==6.3.1
  gguf==0.19.0
  google-auth==2.56.3
  groovy==0.1.2
  h11==0.16.0
  hf-xet==1.6.0
  httpcore==1.0.9
  httptools==0.8.0
  httpx==0.28.1
  huggingface_hub==0.36.2
  idna==3.18
  ImageIO==2.37.4
  importlib_metadata==8.9.0
  interegular==0.3.3
  ipdb==0.13.13
  ipython==9.16.1
  ipython_pygments_lexers==1.1.1
  jedi==0.20.0
  jieba==0.42.1
  Jinja2==3.1.6
  jiter==0.16.0
  joblib==1.5.3
  json_repair==0.63.0
  jsonschema==4.26.0
  jsonschema-specifications==2025.9.1
  kiwisolver==1.5.0
  lark==1.2.2
  latex2sympy2_extended==1.11.0
  lazy-loader==0.5
  Levenshtein==0.27.4
  lightning-utilities==0.15.3
  litellm==1.84.10
  llguidance==0.7.30
  llvmlite==0.44.0
  lm-format-enforcer==0.11.3
  lpips==0.1.4
  lxml==6.1.1
  markdown-it-py==4.2.0
  MarkupSafe==3.0.3
  math-verify==0.9.0
  matplotlib==3.11.1
  matplotlib-inline==0.2.2
  mdurl==0.1.2
  mistral_common==1.11.7
  mpmath==1.3.0
  msgpack==1.2.1
  msgspec==0.21.1
  multidict==6.7.1
  multiprocess==0.70.19
  narwhals==2.24.0
  nest-asyncio==1.6.0
  networkx==3.6.1
  ninja==1.13.0
  nltk==3.10.2
  numba==0.61.2
  numpy==2.2.6
  nvidia-cublas-cu12==12.8.4.1
  nvidia-cuda-cupti-cu12==12.8.90
  nvidia-cuda-nvrtc-cu12==12.8.93
  nvidia-cuda-runtime-cu12==12.8.90
  nvidia-cudnn-cu12==9.10.2.21
  nvidia-cufft-cu12==11.3.3.83
  nvidia-cufile-cu12==1.13.1.3
  nvidia-curand-cu12==10.3.9.90
  nvidia-cusolver-cu12==11.7.3.90
  nvidia-cusparse-cu12==12.5.8.93
  nvidia-cusparselt-cu12==0.7.1
  nvidia-nccl-cu12==2.27.3
  nvidia-nvjitlink-cu12==12.8.93
  nvidia-nvtx-cu12==12.8.90
  omegaconf==2.4.0.dev15
  openai==2.54.0
  openai-clip==1.0.1
  openai-harmony==0.0.8
  opencv-python==5.0.0.93
  opencv-python-headless==5.0.0.93
  openpyxl==3.1.5
  orjson==3.11.9
  outlines_core==0.2.11
  packaging==26.2
  pandas==2.3.3
  parso==0.8.7
  partial-json-parser==0.2.1.1.post7
  pdf2image==1.17.0
  pexpect==4.9.0
  pillow==11.3.0
  Polygon3==3.0.9.1
  portalocker==4.1.0
  prometheus_client==0.26.0
  prompt_toolkit==3.0.53
  propcache==0.5.2
  protobuf==7.35.1
  psutil==7.2.2
  ptyprocess==0.7.0
  pure_eval==0.2.3
  py-cpuinfo==9.0.0
  pyarrow==21.0.0
  pyasn1==0.6.4
  pyasn1_modules==0.4.2
  pybase64==1.5.0
  pycountry==26.2.16
  pycparser==3.0
  pydantic-extra-types==2.11.1
  pydantic-settings==2.15.0
  pydub==0.25.1
  Pygments==2.20.0
  pylatexenc==2.10
  pyparsing==3.3.2
  pypdf==6.15.0
  python-dateutil==2.9.0.post0
  python-dotenv==1.2.2
  python-json-logger==4.1.0
  python-multipart==0.0.32
  pytz==2026.3.post1
  PyYAML==6.0.3
  pyzmq==27.1.0
  qwen-vl-utils==0.0.14
  RapidFuzz==3.14.5
  ray==2.57.0
  referencing==0.37.0
  regex==2026.7.19
  requests==2.34.2
  rich==15.0.0
  rich-toolkit==0.20.3
  rignore==0.8.1
  rouge==1.0.1
  rouge_score==0.1.2
  rpds-py==2026.6.3
  ruff==0.16.2
  safehttpx==0.1.7
  safetensors==0.8.0
  scikit-image==0.26.0
  scikit-learn==1.9.0
  scipy==1.18.0
  semantic-version==2.10.0
  sentence-transformers==5.7.0
  sentencepiece==0.2.2
  sentry-sdk==2.67.1
  setproctitle==1.3.7
  setuptools==79.0.1
  shellingham==1.5.4
  six==1.17.0
  sniffio==1.3.1
  soundfile==0.14.0
  soxr==1.1.0
  stack-data==0.6.3
  sty==1.0.6
  sympy==1.14.0
  tabulate==0.10.0
  tenacity==9.1.4
  termcolor==3.3.0
  threadpoolctl==3.6.0
  tifffile==2026.7.31
  tiktoken==0.13.0
  timeout-decorator==0.5.0
  timm==1.0.28
  tinycss2==1.5.1
  tokenizers==0.22.2
  tomlkit==0.13.3
  torchmetrics==1.9.0
  tqdm==4.70.0
  traitlets==5.16.1
  transformers==4.57.3
  triton==3.4.0
  typer==0.27.1
  typing-inspection==0.4.3
  typing_extensions==4.16.0
  tzdata==2026.3
  urllib3==2.7.0
  uvicorn==0.52.1
  uvloop==0.22.1
  validators==0.35.0
  vllm==0.10.2
  watchfiles==1.2.0
  wcwidth==0.8.2
  webencodings==0.5.1
  websockets==15.0.1
  wheel==0.47.0
  xformers==0.0.32.post1
  xgrammar==0.1.23
  xlsxwriter==3.2.9
  xxhash==3.8.1
  yarl==1.24.5
  zipp==4.1.0
  zss==1.2.0
)

conda run -n "${ENV_NAME}" python -m pip install "${PACKAGES[@]}"
conda run -n "${ENV_NAME}" python -m pip install -e "${VLM_EVAL_PATH}"
conda run -n "${ENV_NAME}" python -m pip install -e "${PROJECT_ROOT}" --no-deps

# Fail on real dependency conflicts. The PyPI metadata of decord 0.6.0 emits
# an incorrect platform warning on this Linux environment even though the
# module imports and is used successfully, so ignore only that exact line.
PIP_CHECK_OUTPUT="$(conda run -n "${ENV_NAME}" python -m pip check 2>&1 || true)"
REAL_CONFLICTS="$(printf '%s\n' "${PIP_CHECK_OUTPUT}" | sed \
    -e '/^decord 0\.6\.0 is not supported on this platform$/d' \
    -e '/^ERROR conda\.cli\.main_run:execute.*python -m pip check.*failed/d' \
    -e '/^[[:space:]]*$/d')"
if [[ -n "${REAL_CONFLICTS}" && "${REAL_CONFLICTS}" != "No broken requirements found." ]]; then
    echo "Dependency verification failed:" >&2
    printf '%s\n' "${REAL_CONFLICTS}" >&2
    exit 3
fi

conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import transformers
import vllm
import vlmeval
import qwenvl

assert torch.__version__.split("+")[0] == "2.8.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda

import torchvision
assert torchvision.__version__.split("+")[0] == "0.23.0", torchvision.__version__

print("Installed eval environment successfully")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
PY
