#!/usr/bin/env bash
# fieldlink-sync.sh — Bidirectional sync with fieldlink peers
#
# Handles both inbound (pull) and outbound (push) sync for all peers
# defined in .fieldlink.json.  Replaces the old unidirectional fieldlink-pull.sh.
#
# Usage:
#   ./fieldlink-sync.sh              # full sync (pull + push)
#   ./fieldlink-sync.sh --pull       # inbound only
#   ./fieldlink-sync.sh --push       # outbound only (generate exports, stage for peers)
#   ./fieldlink-sync.sh --status     # show sync state without changing anything
#   ./fieldlink-sync.sh --peer NAME  # sync only the named peer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/.fieldlink.json"
CACHE_DIR="$SCRIPT_DIR/.fieldlink"
STAGE_DIR="$CACHE_DIR/merge_stage"
MANIFEST_DIR="$SCRIPT_DIR/atlas/exports"

MODE="full"   # full | pull | push | status
PEER_FILTER=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull)   MODE="pull";   shift ;;
    --push)   MODE="push";   shift ;;
    --status) MODE="status"; shift ;;
    --peer)   PEER_FILTER="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--pull|--push|--status] [--peer NAME]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: .fieldlink.json not found at $CONFIG"
  exit 1
fi

if ! command -v jq &> /dev/null; then
  echo "ERROR: jq is required. Install with: apt install jq / brew install jq"
  exit 1
fi

mkdir -p "$STAGE_DIR" "$MANIFEST_DIR"

VERSION=$(jq -r '.fieldlink.version' "$CONFIG")
IDENTITY_NAME=$(jq -r '.fieldlink.identity.name // "unknown"' "$CONFIG")

echo "=== Fieldlink Sync v${VERSION} (${IDENTITY_NAME}) ==="
echo "Config: $CONFIG"
echo "Mode:   $MODE"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
compute_sha256() {
  if command -v sha256sum &> /dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum &> /dev/null; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    echo "no-hash-tool"
  fi
}

resolve_conflict() {
  local file_path="$1"
  local peer_name="$2"
  local config="$3"

  # Check conflict rules for this peer
  local authority
  authority=$(jq -r --arg path "$file_path" --arg peer "$peer_name" '
    .fieldlink.peers[]
    | select(.name == $peer)
    | .sync.conflict_rules[]
    | select($path | test(.pattern | gsub("\\*\\*"; ".*") | gsub("\\*"; "[^/]*")))
    | .authority
  ' "$config" 2>/dev/null | head -1)

  if [ -z "$authority" ] || [ "$authority" = "null" ]; then
    # Fall back to default conflict resolution
    authority=$(jq -r --arg peer "$peer_name" '
      .fieldlink.peers[] | select(.name == $peer) | .sync.conflict_resolution
    ' "$config" 2>/dev/null)
  fi

  echo "$authority"
}

# ---------------------------------------------------------------------------
# Inbound: Pull from peers
# ---------------------------------------------------------------------------
do_pull() {
  echo "--- INBOUND (pull) ---"
  echo ""

  local peers
  peers=$(jq -r '.fieldlink.peers[] | @base64' "$CONFIG")

  for row in $peers; do
    _jq() { echo "$row" | base64 -d | jq -r "${1}"; }

    local name direction repo ref
    name=$(_jq '.name')
    direction=$(_jq '.direction // "inbound"')
    repo=$(_jq '.repo')
    ref=$(_jq '.ref')

    if [ -n "$PEER_FILTER" ] && [ "$name" != "$PEER_FILTER" ]; then
      continue
    fi

    if [ "$direction" != "bidirectional" ] && [ "$direction" != "inbound" ]; then
      echo "  [$name] Skipping (direction=$direction, not inbound)"
      continue
    fi

    echo "  [$name] Pulling from $repo @ $ref"

    local clone_dir="$CACHE_DIR/repos/$name"

    if [ -d "$clone_dir/.git" ]; then
      echo "  [$name] Updating existing clone..."
      git -C "$clone_dir" fetch origin "$ref" --depth 1 2>/dev/null || {
        echo "  [$name] WARNING: Fetch failed (network?). Using cached version."
        continue
      }
      git -C "$clone_dir" checkout "origin/$ref" -- . 2>/dev/null || true
    else
      echo "  [$name] Cloning (shallow)..."
      mkdir -p "$clone_dir"
      git clone --depth 1 --branch "$ref" "$repo" "$clone_dir" 2>/dev/null || {
        echo "  [$name] WARNING: Clone failed. Skipping."
        continue
      }
    fi

    # Process mounts from legacy sources (backward compat)
    local legacy_mounts
    legacy_mounts=$(jq -c --arg name "$name" '
      .fieldlink.sources[] | select(.name == $name) | .mounts // []
    ' "$CONFIG" 2>/dev/null || echo "[]")

    if [ "$legacy_mounts" != "[]" ] && [ "$legacy_mounts" != "null" ] && [ -n "$legacy_mounts" ]; then
      echo "$legacy_mounts" | jq -c '.[]' | while read -r mount; do
        local remote as_path src dest
        remote=$(echo "$mount" | jq -r '.remote')
        as_path=$(echo "$mount" | jq -r '.as')
        src="$clone_dir/$remote"
        dest="$STAGE_DIR/$as_path"
        if [ -f "$src" ]; then
          mkdir -p "$(dirname "$dest")"
          cp "$src" "$dest"
          echo "  [$name] Mounted: $remote -> $as_path"
        else
          echo "  [$name] WARNING: Mount source not found: $remote"
        fi
      done
    fi

    # Process inbound paths from peer config
    local mount_root
    mount_root=$(_jq '.inbound.mount_root // ""')
    if [ -n "$mount_root" ] && [ "$mount_root" != "null" ]; then
      local inbound_paths
      inbound_paths=$(echo "$row" | base64 -d | jq -r '.inbound.paths[]' 2>/dev/null || true)
      for pattern in $inbound_paths; do
        # Expand glob patterns from the clone dir
        local matched_files
        matched_files=$(cd "$clone_dir" && find . -path "./$pattern" -type f 2>/dev/null || true)
        if [ -z "$matched_files" ]; then
          # Try without glob (direct file)
          if [ -f "$clone_dir/$pattern" ]; then
            matched_files="./$pattern"
          fi
        fi
        for match in $matched_files; do
          local rel="${match#./}"
          local dest="$STAGE_DIR/$mount_root/$rel"
          mkdir -p "$(dirname "$dest")"
          cp "$clone_dir/$rel" "$dest"
        done
      done
    fi

    # Record sync state
    local sync_record="$CACHE_DIR/sync_state_${name}.json"
    jq -n \
      --arg peer "$name" \
      --arg direction "inbound" \
      --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --arg ref "$ref" \
      '{peer: $peer, direction: $direction, last_sync: $timestamp, ref: $ref, status: "ok"}' \
      > "$sync_record"

    echo "  [$name] Inbound sync complete."
    echo ""
  done
}

# ---------------------------------------------------------------------------
# Outbound: Generate exports and stage for peers
# ---------------------------------------------------------------------------
do_push() {
  echo "--- OUTBOUND (push) ---"
  echo ""

  # Step 1: Run export generators
  local export_script="$SCRIPT_DIR/scripts/fieldlink_export.py"
  if [ -f "$export_script" ]; then
    echo "  Generating exports..."
    python "$export_script" --output-dir "$MANIFEST_DIR" || {
      echo "  WARNING: Export generation failed."
    }
    echo ""
  fi

  # Step 2: Stage outbound files for each peer
  local peers
  peers=$(jq -r '.fieldlink.peers[] | @base64' "$CONFIG")

  for row in $peers; do
    _jq() { echo "$row" | base64 -d | jq -r "${1}"; }

    local name direction
    name=$(_jq '.name')
    direction=$(_jq '.direction // "inbound"')

    if [ -n "$PEER_FILTER" ] && [ "$name" != "$PEER_FILTER" ]; then
      continue
    fi

    if [ "$direction" != "bidirectional" ] && [ "$direction" != "outbound" ]; then
      echo "  [$name] Skipping outbound (direction=$direction)"
      continue
    fi

    echo "  [$name] Staging outbound files..."

    local outbound_dir="$CACHE_DIR/outbound/$name"
    mkdir -p "$outbound_dir"

    local outbound_paths
    outbound_paths=$(echo "$row" | base64 -d | jq -r '.outbound.paths[]' 2>/dev/null || true)

    local file_count=0
    for path in $outbound_paths; do
      if [ -f "$SCRIPT_DIR/$path" ]; then
        mkdir -p "$outbound_dir/$(dirname "$path")"
        cp "$SCRIPT_DIR/$path" "$outbound_dir/$path"
        local hash
        hash=$(compute_sha256 "$SCRIPT_DIR/$path")
        echo "  [$name] Staged: $path (sha256:${hash:0:12}...)"
        file_count=$((file_count + 1))
      else
        echo "  [$name] WARNING: Outbound file not found: $path"
      fi
    done

    # Write outbound manifest
    local outbound_manifest="$outbound_dir/_outbound_manifest.json"
    jq -n \
      --arg from "$IDENTITY_NAME" \
      --arg to "$name" \
      --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson count "$file_count" \
      '{
        from: $from,
        to: $to,
        generated_at: $timestamp,
        file_count: $count,
        status: "staged",
        instruction: "Copy these files into your atlas/remote/\($from)/ directory"
      }' \
      > "$outbound_manifest"

    # Record sync state
    local sync_record="$CACHE_DIR/sync_state_${name}_outbound.json"
    jq -n \
      --arg peer "$name" \
      --arg direction "outbound" \
      --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson count "$file_count" \
      '{peer: $peer, direction: $direction, last_sync: $timestamp, files_staged: $count, status: "ok"}' \
      > "$sync_record"

    echo "  [$name] Outbound: $file_count files staged at $outbound_dir"
    echo ""
  done
}

# ---------------------------------------------------------------------------
# Status: Show sync state
# ---------------------------------------------------------------------------
do_status() {
  echo "--- SYNC STATUS ---"
  echo ""

  echo "Peers:"
  jq -r '.fieldlink.peers[] | "  \(.name): \(.direction) (\(.repo))"' "$CONFIG"
  echo ""

  echo "Sync records:"
  for record in "$CACHE_DIR"/sync_state_*.json; do
    if [ -f "$record" ]; then
      jq -r '"  \(.peer) [\(.direction)]: \(.status) @ \(.last_sync)"' "$record"
    fi
  done
  if ! ls "$CACHE_DIR"/sync_state_*.json &>/dev/null; then
    echo "  No sync records found. Run a sync first."
  fi
  echo ""

  echo "Staged inbound files:"
  find "$STAGE_DIR" -type f 2>/dev/null | sort | sed 's|^|  |'
  echo ""

  echo "Staged outbound files:"
  for peer_dir in "$CACHE_DIR"/outbound/*/; do
    if [ -d "$peer_dir" ]; then
      local peer_name
      peer_name=$(basename "$peer_dir")
      echo "  [$peer_name]:"
      find "$peer_dir" -type f 2>/dev/null | sort | sed "s|^|    |"
    fi
  done
  if ! ls -d "$CACHE_DIR"/outbound/*/ &>/dev/null 2>&1; then
    echo "  No outbound files staged. Run --push first."
  fi
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
  full)
    do_pull
    do_push
    ;;
  pull)
    do_pull
    ;;
  push)
    do_push
    ;;
  status)
    do_status
    ;;
esac

echo ""
echo "=== Fieldlink sync complete ==="
