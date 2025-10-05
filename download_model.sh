#!/bin/bash

# This script downloads and unpacks the PulsePPG model weights.

# Set the URL and the desired output filename
URL="https://zenodo.org/records/17271045/files/pulseppg_model_weights.zip?download=1"
FILENAME="pulseppg_model_weights.zip"

# --- Download the file ---
echo "Downloading PulsePPG model weights..."
wget -O "$FILENAME" "$URL"

# Check if the download was successful
if [ $? -ne 0 ]; then
  echo "Error: Download failed. Please check the URL and your internet connection."
  exit 1
fi

echo "Download complete."

# --- Unpack the zip file ---
echo "Unpacking the zip file..."
unzip "$FILENAME"

# Check if unpacking was successful
if [ $? -ne 0 ]; then
  echo "Error: Unpacking failed. The file might be corrupted or you may not have 'unzip' installed."
  exit 1
fi

echo "Unpacking complete."
echo "Cleaning up the downloaded zip file..."
rm "$FILENAME"

echo "Done! The model weights have been downloaded and unpacked."