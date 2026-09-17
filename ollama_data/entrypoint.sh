#!/bin/sh
# Start ollama in the background, pull the models the stack is configured for,
# then stay in the foreground on the server process.
#
# Two models, because the two jobs pull in opposite directions: the parser
# wants the strongest model that fits in local RAM, while the judge runs on
# every load of the /parser page and wants the smallest one that can still
# give a verdict. Both names come from the environment so the pull, the
# healthcheck and the backend always agree on what is meant; changing them in
# .env is enough.
MODEL="${OLLAMA_MODEL:-qwen3.5:4b}"
JUDGE_MODEL="${OLLAMA_JUDGE_MODEL:-llama3.2:3b}"

ollama serve &

until ollama list >/dev/null 2>&1; do
  sleep 1
done

for name in "$MODEL" "$JUDGE_MODEL"; do
  echo "[entrypoint] pulling $name"
  ollama pull "$name"
done

wait
