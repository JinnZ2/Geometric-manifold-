#!/usr/bin/env bash
# fieldlink-pull.sh — Pull and stage sources defined in .fieldlink.json
#
# Mirrors the Rosetta-Shape-Core fieldlink pattern.
# Clones/updates configured sibling repos into .fieldlink/merge_stage/.
#
# Usage: ./fieldlink-pull.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/.fieldlink.json"
CACHE_DIR="$SCRIPT_DIR/.fieldlink"
STAGE_DIR="$CACHE_DIR/merge_stage"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: .fieldlink.json not found at $CONFIG"
  exit 1
fi

if ! command -v jq &> /dev/null; then
  echo "ERROR: jq is required. Install with: apt install jq / brew install jq"
  exit 1
fi

mkdir -p "$STAGE_DIR"

echo "=== Fieldlink Pull (Geometric-manifold-) ==="
echo "Config: $CONFIG"
echo "Stage:  $STAGE_DIR"
echo ""

SOURCES=$(jq -r '.fieldlink.sources[] | @base64' "$CONFIG")

for row in $SOURCES; do
  _jq() { echo "$row" | base64 -d | jq -r "${1}"; }

  NAME=$(_jq '.name')
  REPO=$(_jq '.repo')
  REF=$(_jq '.ref')

  CLONE_DIR="$CACHE_DIR/repos/$NAME"

  echo "--- Source: $NAME ---"
  echo "  Repo: $REPO"
  echo "  Ref:  $REF"

  if [ -d "$CLONE_DIR/.git" ]; then
    echo "  Updating existing clone..."
    git -C "$CLONE_DIR" fetch origin "$REF" --depth 1 2>/dev/null || {
      echo "  WARNING: Failed to fetch $NAME (network issue?). Using cached version."
      continue
    }
    git -C "$CLONE_DIR" checkout "origin/$REF" -- . 2>/dev/null || true
  else
    echo "  Cloning (shallow)..."
    mkdir -p "$CLONE_DIR"
    git clone --depth 1 --branch "$REF" "$REPO" "$CLONE_DIR" 2>/dev/null || {
      echo "  WARNING: Failed to clone $NAME. Skipping."
      continue
    }
  fi

  # Process mounts
  MOUNTS=$(echo "$row" | base64 -d | jq -c '.mounts // []' 2>/dev/null)
  if [ "$MOUNTS" != "[]" ] && [ "$MOUNTS" != "null" ]; then
    echo "$MOUNTS" | jq -c '.[]' | while read -r mount; do
      REMOTE=$(echo "$mount" | jq -r '.remote')
      AS=$(echo "$mount" | jq -r '.as')
      SRC="$CLONE_DIR/$REMOTE"
      DEST="$STAGE_DIR/$AS"
      if [ -f "$SRC" ]; then
        mkdir -p "$(dirname "$DEST")"
        cp "$SRC" "$DEST"
        echo "  Mounted: $REMOTE -> $AS"
      else
        echo "  WARNING: Mount source not found: $REMOTE"
      fi
    done
  fi

  echo ""
done

# Run exports if generator exists
EXPORT_SCRIPT="$SCRIPT_DIR/scripts/fieldlink_export.py"
if [ -f "$EXPORT_SCRIPT" ]; then
  echo "--- Generating exports ---"
  python "$EXPORT_SCRIPT" || echo "WARNING: Export generation failed."
  echo ""
fi

echo "=== Fieldlink pull complete ==="
echo "Staged files:"
find "$STAGE_DIR" -type f 2>/dev/null | sort | sed 's|^|  |'
