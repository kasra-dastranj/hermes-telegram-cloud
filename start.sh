#!/bin/sh
set -eu

require_env() {
    name="$1"
    eval "value=\${$name:-}"
    if [ -z "$value" ]; then
        echo "[startup] Missing required deployment secret: $name" >&2
        exit 1
    fi
}

require_env TELEGRAM_BOT_TOKEN
require_env TELEGRAM_ALLOWED_USERS

export HERMES_HOME="${HERMES_HOME:-/opt/data}"
export DATA_DIR="${DATA_DIR:-/opt/data/9router}"
export PUBLIC_PORT="${PORT:-7860}"
export TELEGRAM_WEBHOOK_PORT="${TELEGRAM_WEBHOOK_PORT:-8443}"
export HERMES_TELEGRAM_DISABLE_FALLBACK_IPS="${HERMES_TELEGRAM_DISABLE_FALLBACK_IPS:-true}"
export HERMES_TELEGRAM_INIT_TIMEOUT="${HERMES_TELEGRAM_INIT_TIMEOUT:-15}"

if [ -z "${TELEGRAM_WEBHOOK_URL:-}" ]; then
    if [ -n "${SPACE_HOST:-}" ]; then
        public_host="$SPACE_HOST"
    elif [ -n "${RENDER_EXTERNAL_HOSTNAME:-}" ]; then
        public_host="$RENDER_EXTERNAL_HOSTNAME"
    else
        echo "[startup] No public hostname detected; set TELEGRAM_WEBHOOK_URL manually." >&2
        exit 1
    fi
    export TELEGRAM_WEBHOOK_URL="https://${public_host}/telegram"
fi

# Telegram accepts a hexadecimal secret token. Deriving it from the private
# bot token avoids storing a second secret while keeping the webhook protected.
if [ -z "${TELEGRAM_WEBHOOK_SECRET:-}" ]; then
    TELEGRAM_WEBHOOK_SECRET="$(printf '%s' "$TELEGRAM_BOT_TOKEN" | sha256sum | cut -d ' ' -f 1)"
    export TELEGRAM_WEBHOOK_SECRET
fi

mkdir -p "$HERMES_HOME" "$DATA_DIR"

if [ -d /opt/9router-seed/runtime ] && [ ! -d "$DATA_DIR/runtime" ]; then
    echo "[startup] Seeding 9Router runtime dependencies..."
    cp -a /opt/9router-seed/. "$DATA_DIR/"
fi

# Recreate only the non-secret Hermes configuration on every cold start.
# Telegram credentials remain environment variables managed by Space Secrets.
cat > "$HERMES_HOME/config.yaml" <<'YAML'
model:
  default: oc/deepseek-v4-flash-free
  provider: custom
  base_url: http://127.0.0.1:20128/v1
  api_key: local-no-key-required
  api_mode: chat_completions
agent:
  max_turns: 60
  verbose: false
  reasoning_effort: medium
terminal:
  backend: local
  cwd: /opt/data/workspace
display:
  compact: false
  streaming: true
stt:
  enabled: false
platforms:
  telegram:
    extra:
      status_indicator: true
YAML

mkdir -p "$HERMES_HOME/workspace"

echo "[startup] Opening public port ${PUBLIC_PORT} immediately..."
python3 /app/front_proxy.py &
proxy_pid=$!

echo "[startup] Starting 9Router on internal port 20128..."
9router --host 127.0.0.1 --port 20128 --no-browser --skip-update --log &
router_pid=$!

cleanup() {
    kill "$proxy_pid" 2>/dev/null || true
    kill "$router_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

attempt=0
until curl -fsS http://127.0.0.1:20128/api/health >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 90 ]; then
        echo "[startup] 9Router did not become ready in time." >&2
        exit 1
    fi
    sleep 2
done

echo "[startup] Configuring 9Router's private local database..."
node /app/configure_9router.js
echo "[startup] OpenCode free model is ready and the internal API accepts Hermes."

echo "[startup] Starting Hermes Telegram webhook at ${TELEGRAM_WEBHOOK_URL}"
telegram_probe_status="$(python3 - <<'PY'
import os
import urllib.error
import urllib.request

url = "https://api.telegram.org/bot" + os.environ["TELEGRAM_BOT_TOKEN"] + "/getMe"
try:
    with urllib.request.urlopen(url, timeout=20) as response:
        print(response.status)
except urllib.error.HTTPError as exc:
    print(exc.code)
except Exception:
    print("000")
PY
)"
echo "[startup] Direct Telegram API probe returned HTTP ${telegram_probe_status:-000}."
exec hermes gateway run
