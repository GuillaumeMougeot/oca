#!/bin/sh
# oca installer: uv (if missing) -> oca -> binaries. The model downloads on first `oca`.
#   curl -LsSf https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh | sh
set -eu
command -v git >/dev/null 2>&1 || { echo "oca: install git first (e.g. sudo apt install git)" >&2; exit 1; }
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
# The newest release; OCA_SOURCE overrides it (e.g. git+https://github.com/GuillaumeMougeot/oca for main).
tag=$(git ls-remote --tags --refs --sort=-v:refname https://github.com/GuillaumeMougeot/oca 'v*' | head -n 1 | sed 's|.*refs/tags/||')
uv tool install --force --python 3.12 "${OCA_SOURCE:-git+https://github.com/GuillaumeMougeot/oca@${tag:-main}}"
oca setup
