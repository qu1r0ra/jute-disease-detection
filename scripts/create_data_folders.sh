#!/bin/bash

# This script creates the directory structure for the jute disease and pest dataset.
# It explicitly creates top-level folders followed by class subfolders.

# Define paths
DISEASE_CLASSES="data/disease_classes.txt"
# PEST_CLASSES="data/pest_classes.txt"
DATA_ROOT="data"

echo "Initializing top-level data directories..."
mkdir -p "$DATA_ROOT/by_class"
mkdir -p "$DATA_ROOT/ml_split/train"
mkdir -p "$DATA_ROOT/ml_split/val"
mkdir -p "$DATA_ROOT/ml_split/test"

# Create directories for each class
cat "$DISEASE_CLASSES" | while read -r line; do
  # Skip empty lines
  [ -z "$line" ] && continue

  echo "Creating subfolders for: $line"

  mkdir -p "$DATA_ROOT/by_class/$line"
  mkdir -p "$DATA_ROOT/ml_split/train/$line"
  mkdir -p "$DATA_ROOT/ml_split/val/$line"
  mkdir -p "$DATA_ROOT/ml_split/test/$line"
done

echo "Data structure creation complete!"
