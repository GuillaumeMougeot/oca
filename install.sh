#!/bin/sh
# oca installer: uv (if missing) -> oca -> binaries. The model downloads on first `oca`.
#   curl -LsSf https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh | sh
set -eu
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv tool install --force --python 3.12 "git+https://github.com/GuillaumeMougeot/oca"
oca setup
