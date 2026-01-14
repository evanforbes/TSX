#!/bin/bash
# Auction House - TSX Stock Scanner Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Auction House..."

# Kill any existing instances
pkill -f "python3 app.py" 2>/dev/null
pkill -f "ngrok http" 2>/dev/null
sleep 1

# Start the web app
echo "Starting web server..."
python3 app.py > /tmp/auctionhouse.log 2>&1 &
sleep 2

# Start ngrok tunnel
echo "Starting public tunnel..."
./ngrok http 8080 > /tmp/ngrok.log 2>&1 &
sleep 3

# Get the public URL
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['tunnels'][0]['public_url'])" 2>/dev/null)

echo ""
echo "✅ Auction House is running!"
echo ""
echo "📱 Public URL (works anywhere):"
echo "   $PUBLIC_URL"
echo ""
echo "🏠 Local URL (same WiFi):"
echo "   http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop"

# Wait for user to stop
wait
