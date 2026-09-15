#!/bin/sh
# Start ollama in the background, pull the model the stack is configured for,
# then stay in the foreground on the server process.
#
# The model comes from OLLAMA_MODEL so the pull, the healthcheck and the
# backend always agree on which one is meant; changing it in .env is enough.
MODEL="${OLLAMA_MODEL:-qwen3.5:4b}"

ollama serve &

until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "[entrypoint] pulling $MODEL"
ollama pull "$MODEL"

wait
