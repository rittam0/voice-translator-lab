#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-$project_dir/.tools/uv}"
venv_dir="${VT_PROTOTYPE_VENV:-$project_dir/.venv-prototype}"
export PATH="$project_dir/.tools:$PATH"

if [[ ! -x "$uv_bin" ]]; then
  echo "uv is missing at $uv_bin"
  echo "Install it there from https://github.com/astral-sh/uv/releases, then rerun."
  exit 2
fi
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "ffmpeg and ffprobe are required. On Ubuntu/WSL:"
  echo "  sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1"
  exit 2
fi

"$uv_bin" python install 3.10
if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$uv_bin" venv --python 3.10 "$venv_dir"
fi
"$uv_bin" pip install --python "$venv_dir/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.2.2 torchaudio==2.2.2
"$uv_bin" pip install --python "$venv_dir/bin/python" \
  -r "$project_dir/requirements-lightweight.txt" \
  -r "$project_dir/requirements-prototype-ml.txt"
"$uv_bin" pip install --python "$venv_dir/bin/python" --no-deps \
  "myshell-openvoice @ git+https://github.com/myshell-ai/OpenVoice.git@74a1d147b17a8c3092dd5430504bd83ef6c7eb23" \
  "melotts @ git+https://github.com/myshell-ai/MeloTTS.git@209145371cff8fc3bd60d7be902ea69cbdb7965a"

echo "Prototype environment ready: $venv_dir"
