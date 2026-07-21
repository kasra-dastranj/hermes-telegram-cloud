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

The default `hermes-free` combo tries these models in order. Provider
connections are recreated from deployment secrets on every cold start:

1. `oc/deepseek-v4-flash-free`
2. `oc/mimo-v2.5-free`
3. `oc/big-pickle`
4. `oc/nemotron-3-ultra-free`
5. `oc/north-mini-code-free`
6. `groq/openai/gpt-oss-120b` (when `GROQ_API_KEY` is set)
7. `groq/llama-3.3-70b-versatile` (when `GROQ_API_KEY` is set)
8. `groq/qwen/qwen3.6-27b` (when `GROQ_API_KEY` is set)
9. `openrouter/openrouter/free` (when `OPENROUTER_API_KEY` is set)

9Router automatically falls back to the next model when the current one is
rate-limited, out of quota, overloaded, or otherwise unavailable.

Incoming Telegram voice messages are transcribed with Groq's
`whisper-large-v3-turbo` model when the `GROQ_API_KEY` deployment secret is
set. Hermes echoes the transcript into the chat before answering it.

Incoming Telegram images are analyzed by OpenRouter's `openrouter/free`
router when the `OPENROUTER_API_KEY` deployment secret is set. Hermes keeps
the text-only 9Router combo as the main agent and injects the vision model's
image description into the conversation.

Required Space secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_USERS`
- `GROQ_API_KEY` (required for Telegram voice transcription)
- `OPENROUTER_API_KEY` (required for Telegram image analysis)

The Telegram webhook URL and webhook secret are derived automatically at
runtime on Render or Hugging Face Spaces. No credentials are stored in this
public repository.

The included `render.yaml` deploys a free Render web service. Free instances
sleep when idle; Telegram webhook traffic wakes the service automatically.
