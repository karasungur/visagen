#!/bin/bash
# Visagen Installation Script (Linux/macOS wrapper)
# Usage: ./install.sh

set -e

# Check Python 3
if command -v python3 &> /dev/null; then
    python3 install.py "$@"
elif command -v python &> /dev/null; then
    python install.py "$@"
else
    echo "Error: Python 3 not found. Please install Python 3.10+"
    exit 1
fi
