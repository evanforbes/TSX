#!/bin/bash
# Start the TSX Stock Scanner web server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing instance
pkill -f "python3 app.py" 2>/dev/null

# Wait a moment
sleep 1

# Start the Flask server
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 app.py
