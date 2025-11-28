#!/bin/bash
# Morgan v2-0.0.1 Startup Script

cd "$(dirname "$0")/morgan-rag"

echo "================================================="
echo "  Morgan v2-0.0.1 - Emotional AI Assistant"
echo "================================================="
echo ""

# Set Python path
export PYTHONPATH=./local_libs:$PYTHONPATH

# Check if quick test passes
echo "Running system check..."
python3 scripts/quick_test.py 2>&1 | grep -E "✅|❌"

echo ""
echo "================================================="
echo ""
echo "Choose an option:"
echo "  1) Interactive Chat (CLI)"
echo "  2) Import Conversations"
echo "  3) Run Tests"
echo "  4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
  1)
    echo "Starting Morgan interactive chat..."
    python3 scripts/run_morgan.py
    ;;
  2)
    echo "Importing conversations..."
    python3 scripts/import_conversations.py ../conversations.json
    ;;
  3)
    echo "Running tests..."
    python3 scripts/quick_test.py
    ;;
  4)
    echo "Goodbye!"
    exit 0
    ;;
  *)
    echo "Invalid choice"
    exit 1
    ;;
esac
