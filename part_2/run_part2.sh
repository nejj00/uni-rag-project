#!/bin/bash

set -e  # Exit on error

BASE_URL="https://lagom.cs.kuleuven.be/cs/courses/irse"
FILES=(
"acl_anthology_queries.json"
"acl_anthology_full.parquet"
"acl_anthology_queries.parquet"
)

echo "Ensuring input directory exists..."
mkdir -p inputs

echo "Downloading files..."

for FILE in "${FILES[@]}"; do
if [ ! -f "inputs/$FILE" ]; then
wget -P inputs "$BASE_URL/$FILE"
else
echo "File already exists: inputs/$FILE"
fi
done

echo "Running main script..."
python3 main.py

echo "Done."
