#!/bin/bash

set -e  # Exit on error

ENV_NAME=".venv"

# Check if virtual environment already exists

if [ -d "$ENV_NAME" ] && [ -f "$ENV_NAME/bin/activate" ]; then
echo "Virtual environment already exists."
else
echo "Creating virtual environment..."
python3 -m venv $ENV_NAME
fi

echo "Creating folder structure..."
mkdir -p part_1/inputs part_1/results
mkdir -p part_2/inputs part_2/results

echo "Activating virtual environment..."
source $ENV_NAME/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# Only install requirements if file exists

if [ -f "requirements.txt" ]; then
echo "Installing requirements..."
pip install -r requirements.txt
else
echo "No requirements.txt found, skipping installation."
fi

echo "Setup complete."
