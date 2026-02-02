#!/bin/bash
# Viterbox TTS Server Script
# Usage:
#   ./run_server.sh         # Run regular TTS server
#   ./run_server.sh --tagged  # Run tagged TTS server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8088

show_help() {
    echo "Viterbox TTS Server"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --tagged     Run the tagged TTS server (supports [silence Ns], [soundtrack], [speaker_id] tags)"
    echo "  --port PORT  Set server port (default: 8088)"
    echo "  --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                   # Start regular TTS server"
    echo "  $0 --tagged          # Start tagged TTS server"
    echo "  $0 --tagged --port 9000  # Start tagged server on port 9000"
}

# Parse arguments
TAGGED=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --tagged)
            TAGGED=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/Scripts/activate" 2>/dev/null || source "$SCRIPT_DIR/venv/bin/activate" 2>/dev/null
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Run the appropriate server
if [ "$TAGGED" = true ]; then
    echo "Starting Tagged TTS Server on port $PORT..."
    python openai_server/tagged_tts_server.py
else
    echo "Starting TTS Server on port $PORT..."
    python openai_server/tts_server.py
fi
