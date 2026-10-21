PORT=30000


echo "Starting Model."
CUDA_VISIBLE_DEVICES=0 vllm serve ornith-ai/Ornith-1.5-35B-A3B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port ${PORT} \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code > ./vllm.log 2>&1 &


TIMEOUT_SECONDS=1200
INTERVAL_SECONDS=1
ELAPSED=0

while [ "$ELAPSED" -lt "$TIMEOUT_SECONDS" ]; do
  if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" | grep -q "^200$"; then
    echo "Model is healthy after ${ELAPSED} seconds."
    exit 0
  fi
  printf "\rWaiting for model to become healthy... (${ELAPSED}s elapsed)"
  sleep "$INTERVAL_SECONDS"
  ELAPSED=$((ELAPSED + INTERVAL_SECONDS))
done

echo "ERROR: model did not become healthy within ${TIMEOUT_SECONDS} seconds." >&2
exit 1