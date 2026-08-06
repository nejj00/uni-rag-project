#!/bin/bash

set -e  # Exit on error

INPUT_DIR="inputs"

echo "Ensuring input directory exists..."
mkdir -p $INPUT_DIR

echo "Downloading dataset..."

FILE1="$INPUT_DIR/irse_documents_2026_recipes.parquet"
FILE2="$INPUT_DIR/irse_queries_2026_recipes.json"

# Download only if files don't already exist

if [ ! -f "$FILE1" ]; then
wget -P $INPUT_DIR https://people.cs.kuleuven.be/~thomas.bauwens/irse_documents_2026_recipes.parquet
else
echo "File already exists: $FILE1"
fi

if [ ! -f "$FILE2" ]; then
wget -P $INPUT_DIR https://people.cs.kuleuven.be/~thomas.bauwens/irse_queries_2026_recipes.json
else
echo "File already exists: $FILE2"
fi

echo "Running main script..."
python3 main.py

echo "Done."
