#!/bin/bash

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/DOT_RAG"

# Start the Flask app
echo "Starting Flask application..."
python startup.py 