#!/bin/sh
# Start ollama in the background, pull the llama3.2:3b model,
# then stay in the foreground on the server process.
ollama serve &

until ollama list >/dev/null 2>&1; do
  sleep 1
done

ollama pull llama3.2:3b

wait