FROM nousresearch/hermes-agent:latest

USER root

# Seed 9Router's runtime dependencies during the image build. The startup
# script copies this seed into Hermes' writable data directory on each fresh
# free-Space runtime.
ENV DATA_DIR=/opt/9router-seed
RUN mkdir -p /opt/9router-seed \
    && npm install --global 9router@0.5.35 \
    && sed -i 's/--max-old-space-size=6144/--max-old-space-size=192/' "$(npm root --global)/9router/cli.js" \
    && npm cache clean --force

COPY --chmod=0755 start.sh /app/start.sh
COPY front_proxy.py /app/front_proxy.py

ENV HERMES_HOME=/opt/data \
    DATA_DIR=/opt/data/9router \
    TELEGRAM_WEBHOOK_PORT=8443 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# Keep the Hermes image's s6 entrypoint. It prepares /opt/data and then runs
# this executable as the unprivileged hermes user.
CMD ["/app/start.sh"]
