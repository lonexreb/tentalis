#!/bin/bash
set -euo pipefail

echo "=== Starting Tentalis demo ==="

# Start all services
echo "Starting Docker Compose services..."
docker compose up -d

# Wait for NATS
echo "Waiting for NATS..."
until curl -sf http://localhost:8222/healthz > /dev/null 2>&1; do
    sleep 1
done
echo "NATS ready."

# Wait for Ollama
echo "Waiting for Ollama..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama ready."

# Pull the model
echo "Pulling qwen2.5:1.5b model (this may take a while on first run)..."
docker compose exec ollama ollama pull qwen2.5:1.5b

# Wait for Bridge
echo "Waiting for Bridge API..."
until curl -sf http://localhost:8100/health > /dev/null 2>&1; do
    sleep 1
done
echo "Bridge ready."

echo ""
echo "=== All services running ==="
echo ""
echo "  NATS:       nats://localhost:4222  (monitoring: http://localhost:8222)"
echo "  Ollama:     http://localhost:11434"
echo "  Bridge API: http://localhost:8100"
echo "  OpenClaw:   http://localhost:3000  (gateway: ws://localhost:18789)"
echo ""
echo "Open http://localhost:3000 and message the Manager agent."
echo ""
echo "To test the Bridge directly:"
echo '  curl -X POST http://localhost:8100/tasks/assign \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"manager_id":"manager-01","task_type":"coding","prompt":"Write a fibonacci function"}'"'"
echo ""
echo "To stop: docker compose down"
