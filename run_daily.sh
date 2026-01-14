#!/bin/bash
# Daily TSX Scanner Script
# Runs the scanner and saves results to a dated file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/reports"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H%M)

# Create reports directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the scanner and save output
cd "$SCRIPT_DIR"
python3 scanner.py > "$OUTPUT_DIR/scan_${DATE}_${TIME}.txt" 2>&1

# Also keep a copy of the latest scan
cp "$OUTPUT_DIR/scan_${DATE}_${TIME}.txt" "$OUTPUT_DIR/latest_scan.txt"

echo "Scan complete: $OUTPUT_DIR/scan_${DATE}_${TIME}.txt"
