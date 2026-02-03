#!/bin/bash

# ArxivMiner Startup Script

# Create data directory
mkdir -p data

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
