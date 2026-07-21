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
- An automatic 9Router fallback chain of OpenCode Free models

The default `hermes-free` combo tries these models in order:

1. `oc/deepseek-v4-flash-free`
2. `oc/mimo-v2.5-free`
3. `oc/big-pickle`
4. `oc/nemotron-3-ultra-free`
5. `oc/north-mini-code-free`

Required Space secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_USERS`

The Telegram webhook URL and webhook secret are derived automatically at
runtime on Render or Hugging Face Spaces. No credentials are stored in this
public repository.

The included `render.yaml` deploys a free Render web service. Free instances
sleep when idle; Telegram webhook traffic wakes the service automatically.
