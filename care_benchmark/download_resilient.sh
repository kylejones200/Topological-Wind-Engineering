#!/bin/bash
# Resilient CARE dataset download with automatic resume.
set -euo pipefail
DEST="/Volumes/NO NAME/Topological-Wind-Engineering/data/care/CARE_To_Compare.zip"
URL="https://zenodo.org/records/15846963/files/CARE_To_Compare.zip"
EXPECTED=5503439673

mkdir -p "$(dirname "$DEST")"
attempt=1
while true; do
  current=0
  if [[ -f "$DEST" ]]; then
    current=$(stat -f%z "$DEST" 2>/dev/null || stat -c%s "$DEST")
  fi
  if [[ "$current" -ge "$EXPECTED" ]]; then
    echo "Download complete ($(python3 -c "print($current/1e9)") GB)"
    break
  fi
  echo "Attempt $attempt: resuming from $(python3 -c "print(round($current/1e9,2))") GB / 5.5 GB"
  curl -L -C - --retry 5 --retry-delay 10 --speed-time 120 --speed-limit 1000 \
    -o "$DEST" "$URL" || true
  attempt=$((attempt + 1))
  sleep 10
done

cd "$(dirname "$DEST")"
if [[ ! -f "CARE_To_Compare/Wind Farm A/event_info.csv" ]]; then
  echo "Extracting archive..."
  unzip -q "$DEST" -d "$(dirname "$DEST")"
fi
echo "CARE dataset ready."
