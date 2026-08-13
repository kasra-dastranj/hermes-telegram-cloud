FROM nousresearch/hermes-agent:v2026.8.3@sha256:c0cab4e3711bcb27a312be1b3776254fc06fd50d5f7a6b8017915fc7171cb39e

USER root

# Seed 9Router's runtime dependencies during the image build. The startup
# script copies this seed into Hermes' writable data directory on each fresh
# free-Space runtime.
ENV DATA_DIR=/opt/9router-seed
RUN mkdir -p /opt/9router-seed \
    && npm install --global 9router@0.5.35 \
    && sed -i 's/--max-old-space-size=6144/--max-old-space-size=192/' "$(npm root --global)/9router/cli.js" \
    && npm cache clean --force \
    && uv pip install --python /opt/hermes/.venv/bin/python --no-cache \
        'huggingface_hub>=0.34,<2' 'cryptography>=44,<47'

# agent-browser does not itself honor PLAYWRIGHT_BROWSERS_PATH when launching.
# Expose the Chromium already shipped by the pinned Hermes image under a
# standard system name so both requirement checks and the actual CLI find it.
RUN browser_binary="$(find /opt/hermes/.playwright -type f -name chrome-headless-shell -perm -111 | head -n 1)" \
    && test -n "$browser_binary" \
    && ln -sf "$browser_binary" /usr/local/bin/chromium

COPY --chmod=0755 start.sh /app/start.sh
COPY front_proxy.py /app/front_proxy.py
COPY model_proxy.py /app/model_proxy.py
COPY configure_9router.js /app/configure_9router.js
COPY patch_hermes_streaming.py /app/patch_hermes_streaming.py
COPY backup_sync.py /app/backup_sync.py

# Hermes v0.20.0 natively resolves stt.groq.language/stt.language. Keep only
# the narrow non-streaming switch needed by the buffered multi-model fallback.
RUN python3 /app/patch_hermes_streaming.py

ENV HERMES_HOME=/opt/data \
    DATA_DIR=/opt/data/9router \
    PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright \
    TELEGRAM_WEBHOOK_PORT=8443 \
    HERMES_UPSTREAM_STREAMING=true \
    BACKUP_ALLOW_MISSING_REMOTE=false \
    HERMES_GATEWAY_NO_SUPERVISE=1 \
    HOME=/opt/data \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# Render only needs one foreground process. Bypass the base image's s6
# lifecycle so Render's port probe cannot lose the public proxy while s6
# transitions between its static services. Hermes runs in foreground mode;
# the shell keeps 9Router and the public proxy as sibling child processes.
ENTRYPOINT ["/app/start.sh"]
CMD []
