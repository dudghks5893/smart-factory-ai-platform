#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "[FAIL] Homebrew is required for the macOS streaming runtime." >&2
    exit 1
  fi

  LIBFFI_PREFIX="$(brew --prefix libffi)"
  GLIB_PREFIX="$(brew --prefix glib)"
  GSTREAMER_PREFIX="$(brew --prefix gstreamer)"

  export PKG_CONFIG_PATH="$LIBFFI_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export CPPFLAGS="-I$LIBFFI_PREFIX/include ${CPPFLAGS:-}"
  export LDFLAGS="-L$LIBFFI_PREFIX/lib ${LDFLAGS:-}"
  export DYLD_LIBRARY_PATH="$GLIB_PREFIX/lib:$GSTREAMER_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"
fi

cd "$ROOT_DIR"
exec uv run --group streaming "$@"
