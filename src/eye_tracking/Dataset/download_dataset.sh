#!/bin/bash
# Download and extract the Columbia Gaze Data Set

set -e

URL="https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/columbia_gaze_data_set.zip"
DATASET_DIR="columbia_gaze_data_set"
EXPECTED_DIR="$DATASET_DIR/Columbia Gaze Data Set"
ZIP_FILE="columbia_gaze_data_set.zip"

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: ./download_dataset.sh [--force]"
    exit 0
fi

if [ -d "$EXPECTED_DIR" ] && [ "$1" != "--force" ] && [ "$1" != "-f" ]; then
    echo "Dataset already exists. Use --force to re-download."
    exit 0
fi

mkdir -p "$DATASET_DIR"

echo "Downloading dataset (~2.3GB)..."
if command -v curl &> /dev/null; then
    curl -L -o "$ZIP_FILE" "$URL"
elif command -v wget &> /dev/null; then
    wget -O "$ZIP_FILE" "$URL"
else
    echo "Error: curl or wget required"
    exit 1
fi

echo "Extracting..."
unzip -q "$ZIP_FILE" -d "$DATASET_DIR"

rm -f "$ZIP_FILE"

if [ -d "$EXPECTED_DIR" ]; then
    echo "Done. Dataset at: $EXPECTED_DIR"
else
    echo "Error: Extraction failed"
    exit 1
fi
