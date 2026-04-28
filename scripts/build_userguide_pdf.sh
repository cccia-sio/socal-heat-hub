#!/usr/bin/env bash
# Render userguide/index.html to a distributable PDF using headless Chrome.
# Output: docs/userguide/user_guide_external.pdf
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$REPO_ROOT/userguide/index.html"
OUTPUT_DIR="$REPO_ROOT/docs/userguide"
OUTPUT="$OUTPUT_DIR/user_guide_external.pdf"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

if [ ! -x "$CHROME" ]; then
  echo "Error: Google Chrome not found at: $CHROME" >&2
  echo "Install Chrome or update CHROME in this script." >&2
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "Error: input HTML not found: $INPUT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Rendering: $INPUT"
echo "Output:    $OUTPUT"

"$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-pdf-header-footer \
  --virtual-time-budget=5000 \
  --print-to-pdf="$OUTPUT" \
  "file://$INPUT" \
  >/dev/null 2>&1

if [ ! -f "$OUTPUT" ]; then
  echo "Error: PDF was not created." >&2
  exit 1
fi

size=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT")
echo "Done. PDF size: ${size} bytes"
