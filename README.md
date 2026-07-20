---
title: Hermes Telegram
emoji: "⚕️"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Hermes Telegram Cloud

This Docker Space runs:

- Hermes Agent as a Telegram webhook bot
- 9Router on an internal-only port
- `oc/deepseek-v4-flash-free` through 9Router

Required Space secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_USERS`

The Telegram webhook URL and webhook secret are derived automatically at
runtime on Render or Hugging Face Spaces. No credentials are stored in this
public repository.

The included `render.yaml` deploys a free Render web service. Free instances
sleep when idle; Telegram webhook traffic wakes the service automatically.
