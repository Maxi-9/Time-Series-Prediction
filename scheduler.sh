#!/usr/bin/env bash
#
# scheduler.sh — run run_all.py daily at buy_time from config.json
#

set -euo pipefail

# Clean exit on Ctrl‑C or kill
trap 'echo "[$(date "+%Y-%m-%d %H:%M:%S")] Scheduler exiting."; exit 0' SIGINT SIGTERM

# CONFIG: first arg or default to ./config.json
CONFIG=${1:-log/config.json}

# Ensure dependencies & config presence
if ! command -v jq >/dev/null; then
  echo "ERROR: jq is required. Install it (e.g. brew install jq or apt-get install jq)" >&2
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: Config file not found: $CONFIG" >&2
  exit 1
fi

# Read buy_time (HH:MM) from config.json
BUY_TIME=$(jq -r '.buy_time // "09:40"' "$CONFIG")
if ! [[ $BUY_TIME =~ ^([0-1][0-9]|2[0-3]):([0-5][0-9])$ ]]; then
  echo "ERROR: buy_time in config.json must be HH:MM (got '$BUY_TIME')" >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scheduler started."
echo "  → Daily run_all.py at $BUY_TIME local time"
echo "  → Using config: $CONFIG"
echo

# Directory of this script, so we can call run_all.py reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while true; do
  # Split HH:MM into hour/min, handle leading zeros
  IFS=: read -r hour minute <<< "$BUY_TIME"
  target_sec=$((10#$hour * 3600 + 10#$minute * 60))

  # Current time in seconds since midnight
  now_h=$(date +%H)
  now_m=$(date +%M)
  now_s=$(date +%S)
  now_sec=$((10#$now_h * 3600 + 10#$now_m * 60 + 10#$now_s))

  # Compute how many seconds until the next run
  delta=$((target_sec - now_sec))
  if (( delta <= 0 )); then
    delta=$((delta + 86400))  # schedule for tomorrow
  fi

  # Sleep until buy_time
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sleeping for $delta seconds until next run at $BUY_TIME."
  sleep "$delta"

  # Invoke your python script
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running run_all.py …"
  if python3 "$SCRIPT_DIR/run_all.py" "$CONFIG"; then
    echo "[$(date '+%H:%M:%S')] run_all.py succeeded."
  else
    echo "[$(date '+%H:%M:%S')] run_all.py FAILED!" >&2
  fi

  # loop and schedule again
done