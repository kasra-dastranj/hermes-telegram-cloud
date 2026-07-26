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

This Docker deployment runs:

- Hermes Agent as a Telegram webhook bot
- 9Router on an internal-only port
- An automatic 9Router fallback chain of OpenCode Free models

The default `hermes-free` combo tries these models in order. Provider
connections are recreated from deployment secrets on every cold start:

1. `oc/nemotron-3-ultra-free`
2. `groq/openai/gpt-oss-120b` (when `GROQ_API_KEY` is set)
3. `oc/north-mini-code-free`
4. `groq/llama-3.3-70b-versatile` (when `GROQ_API_KEY` is set)
5. `oc/deepseek-v4-flash-free`
6. `groq/qwen/qwen3.6-27b` (when `GROQ_API_KEY` is set)
7. `oc/mimo-v2.5-free`
8. `oc/big-pickle`
9. `openrouter/openrouter/free` (when `OPENROUTER_API_KEY` is set)

The internal model adapter calls the enabled models in order and advances when
the current model is rate-limited, out of quota, overloaded, times out, or
returns an empty/malformed HTTP-200 response. This compensates for providers
that incorrectly report an empty completion as successful.

The adapter owns the complete retry/fallback chain, so Hermes does not repeat
all nine models after a terminal failure. New Telegram messages are queued
while a long task is running, and context is compressed early enough for a
512 MB free instance.
9Router responses are buffered before being converted back to a standards-
compliant OpenAI stream, so a provider can fail over before Telegram delivery
begins. The container supervises Hermes, 9Router, the internal model adapter,
and the public webhook proxy; a failure in any one causes a clean restart.

Incoming Telegram voice messages are transcribed with Groq's
`whisper-large-v3-turbo` model when the `GROQ_API_KEY` deployment secret is
set. Persian (`fa`) is forced by default to prevent short voice notes from
being auto-detected or rendered in English. Override `STT_GROQ_LANGUAGE` only
if a different spoken language is needed. Hermes echoes the transcript into
the chat before answering it.

Incoming Telegram images are analyzed by OpenRouter's `openrouter/free`
router when the `OPENROUTER_API_KEY` deployment secret is set. Hermes keeps
the text-only 9Router combo as the main agent and injects the vision model's
image description into the conversation.

Required deployment secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_USERS`
- `GROQ_API_KEY` (required for Telegram voice transcription)
- `OPENROUTER_API_KEY` (required for Telegram image analysis)
- `HF_TOKEN`: a Hugging Face token with write access to the backup Dataset
- `HF_BACKUP_REPO`: private Dataset ID, for example `username/hermes-backup`
- `BACKUP_ENCRYPTION_KEY`: a separately saved random secret of at least 24 characters

The deployment stores an AES-256-GCM encrypted, allow-listed snapshot after
the first three minutes and every ten minutes thereafter, keeps a bounded
remote rollback history, and restores the latest snapshot after an ephemeral
restart. A restore error stops startup instead of allowing an empty instance
to overwrite the last good backup. This deployment also requires the existing
remote snapshot to be accessible; a missing/inaccessible backup fails closed
instead of silently starting with no memory. Included state is `state.db`,
sessions, memories, cron data, and the workspace. Environment files,
credentials, configuration, 9Router's database, Git metadata, virtual
environments, and obvious key files are excluded. The encryption key is never
uploaded to Hugging Face.

`/health` is a deep readiness check for Hermes, 9Router, and the internal model
adapter. `/` remains a lightweight wake endpoint for Render.

The Telegram webhook URL and webhook secret are derived automatically at
runtime on Render or Hugging Face Spaces. No credentials are stored in this
public repository.

The included `render.yaml` deploys a free Render web service. Free instances
sleep when idle; Telegram webhook traffic wakes the service automatically.
