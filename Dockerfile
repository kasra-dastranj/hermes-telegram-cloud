FROM nousresearch/hermes-agent:v2026.7.20@sha256:f7b35053268f532f98955195c909f15a230470fbcbdacaa9fdecb95707dad04a

USER root

# Seed 9Router's runtime dependencies during the image build. The startup
# script copies this seed into Hermes' writable data directory on each fresh
# free-Space runtime.
ENV DATA_DIR=/opt/9router-seed
RUN mkdir -p /opt/9router-seed \
    && npm install --global 9router@0.5.35 \
    && sed -i 's/--max-old-space-size=6144/--max-old-space-size=192/' "$(npm root --global)/9router/cli.js" \
    && npm cache clean --force \
    && python3 -m pip install --no-cache-dir 'huggingface_hub>=0.34,<2' 'cryptography>=44,<47'

COPY --chmod=0755 start.sh /app/start.sh
COPY front_proxy.py /app/front_proxy.py
COPY configure_9router.js /app/configure_9router.js
COPY patch_hermes_stt.py /app/patch_hermes_stt.py
COPY backup_sync.py /app/backup_sync.py

# Hermes v0.19.0 does not pass a language to Groq Whisper. Patch only that
# exact call at build time so Persian voice notes stay Persian. The patch
# deliberately fails the build if an upstream image changes the target code.
RUN python3 /app/patch_hermes_stt.py

ENV HERMES_HOME=/opt/data \
    DATA_DIR=/opt/data/9router \
    TELEGRAM_WEBHOOK_PORT=8443 \
    STT_GROQ_LANGUAGE=fa \
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
