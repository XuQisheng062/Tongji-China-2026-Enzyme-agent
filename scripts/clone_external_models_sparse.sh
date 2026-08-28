#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"
mkdir -p "$EXT"

clone_sparse() {
  local url="$1" dest="$2" paths="$3"
  if [[ -d "$dest/.git" ]]; then
    echo "[SKIP] \u5df2\u5b58\u5728: $dest"
    return
  fi
  git clone --depth 1 --filter=blob:none --sparse "$url" "$dest"
  # shellcheck disable=SC2086
  git -C "$dest" sparse-checkout set $paths
}

# \u53ea\u53d6\u5b9e\u9645\u63a8\u7406\u9700\u8981\u7684\u4ee3\u7801\u76ee\u5f55\u3002cone \u6a21\u5f0f\u4f1a\u540c\u65f6\u4fdd\u7559\u4ed3\u5e93\u6839\u76ee\u5f55\u6587\u4ef6\uff08README/env/requirements/LICENSE \u7b49\uff09\u3002
clone_sparse "https://github.com/jafetgado/EpHod.git" \
  "$EXT/EpHod" "ephod"

clone_sparse "https://github.com/DeepBxM/EnzGFM.git" \
  "$EXT/EnzGFM" "point_mutation models EsmTokenizer"

clone_sparse "https://github.com/xlab-BioAI/UniStab.git" \
  "$EXT/UniStab" "src datasets utils model config"

echo
echo "\u5916\u90e8\u6a21\u578b\u4ee3\u7801 sparse clone \u5b8c\u6210\uff1a"
du -sh "$EXT"/* || true
