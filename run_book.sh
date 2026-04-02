#!/usr/bin/env bash
# run_book.sh — Build and open the Applied DS Book locally.
#
# Usage:
#   ./run_book.sh           # incremental build (uses cached outputs)
#   ./run_book.sh --clean   # clean build (clears all cached notebook outputs)
#
# Run from the repository root (applied_ds_book/).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOK_DIR="$REPO_ROOT/applied_data_science_book"
VENV_DIR="$REPO_ROOT/adsb_env"
INDEX_HTML="$BOOK_DIR/_build/html/index.html"

# ── 1. Activate virtual environment ──────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR."
    echo "Creating it now..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 2. Install / sync dependencies ───────────────────────────────────────────
echo "Installing dependencies..."
pip install -q -r "$BOOK_DIR/requirements.txt"

# ── 3. Optionally clean previous build ───────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning previous build..."
    jupyter-book clean "$BOOK_DIR"
fi

# ── 4. Build the book ────────────────────────────────────────────────────────
echo "Building book..."
jupyter-book build "$BOOK_DIR"

# ── 5. Open in browser ───────────────────────────────────────────────────────
echo "Opening $INDEX_HTML"
if command -v open &>/dev/null; then
    open "$INDEX_HTML"                  # macOS
elif command -v xdg-open &>/dev/null; then
    xdg-open "$INDEX_HTML"             # Linux
elif command -v start &>/dev/null; then
    start "$INDEX_HTML"                # Windows (Git Bash)
else
    echo "Could not detect a browser launcher. Open manually:"
    echo "  $INDEX_HTML"
fi
