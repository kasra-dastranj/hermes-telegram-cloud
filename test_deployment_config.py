from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def generated_config():
    source = (ROOT / "start.sh").read_text(encoding="utf-8")
    start = source.index("<<'YAML'\n") + len("<<'YAML'\n")
    end = source.index("\nYAML", start)
    return yaml.safe_load(source[start:end])


def test_gateway_and_model_transport_are_deliberately_separate():
    config = generated_config()
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert config["model"]["base_url"] == "http://127.0.0.1:20129/v1"
    assert config["streaming"]["enabled"] is False
    assert config["display"]["streaming"] is False
    assert "HERMES_UPSTREAM_STREAMING=true" in dockerfile
    assert "PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright" in dockerfile
    assert "BACKUP_ALLOW_MISSING_REMOTE=false" in dockerfile
    assert "nousresearch/hermes-agent:v2026.8.3@sha256:" in dockerfile
    assert "patch_hermes_stt.py" not in dockerfile
    assert "fallback-models.json" in (ROOT / "configure_9router.js").read_text(
        encoding="utf-8"
    )
    assert "fallback-models.json" in (ROOT / "model_proxy.py").read_text(
        encoding="utf-8"
    )


def test_fallback_excludes_known_broken_or_stalling_models():
    source = (ROOT / "configure_9router.js").read_text(encoding="utf-8")
    preferred = source.split("const preferredModelOrder = [", 1)[1].split(
        "];", 1
    )[0]

    assert preferred.index('"gorouter/claude-opus-5"') < preferred.index(
        '"groq/openai/gpt-oss-120b"'
    )
    assert "gorouter/claude-opus-5-thinking" not in preferred
    assert "groq/openai/gpt-oss-120b" in preferred
    assert "groq/openai/gpt-oss-20b" in preferred
    assert "groq/qwen/qwen3.6-27b" in preferred
    assert "groq/llama-3.3-70b-versatile" not in preferred
    assert "oc/deepseek-v4-flash-free" not in preferred
    assert "oc/nemotron-3-ultra-free" not in preferred
    assert "oc/north-mini-code-free" not in preferred


def test_long_tasks_and_context_have_defensive_settings():
    config = generated_config()

    assert config["agent"]["intent_ack_continuation"] is True
    assert config["agent"]["api_max_retries"] == 1
    assert config["compression"]["in_place"] is True
    assert config["model"]["context_length"] == 131072
    assert config["model"]["max_tokens"] == 2048
    assert config["compression"]["threshold"] == 0.75
    assert "threshold_tokens" not in config["compression"]
    assert config["compression"]["target_ratio"] == 0.45
    assert config["compression"]["protect_last_n"] == 12
    assert config["compression"]["proactive_prune_tokens"] == 40000
    assert config["compression"]["idle_compact_after_seconds"] == 1800
    assert config["stt"]["language"] == "fa"
    assert config["stt"]["groq"]["language"] == "fa"


def test_telegram_inherits_the_official_full_default_toolset():
    config = generated_config()

    assert "platform_toolsets" not in config
    assert "disabled_toolsets" not in config["agent"]


def test_personal_assistant_bootstrap_uses_memory_and_tools_without_overwrite():
    source = (ROOT / "start.sh").read_text(encoding="utf-8")

    assert 'if [ ! -e "$agent_rules" ]' in source
    assert "use `memory` or `session_search`" in source
    assert "proactively use the" in source
    assert "Use cronjob for scheduling requests" in source


def test_startup_fails_closed_on_backup_restore_errors():
    source = (ROOT / "start.sh").read_text(encoding="utf-8")

    assert "backup_sync.py restore || true" not in source
    assert "if [ \"$restore_status\" -eq 1 ]" in source
    for name in ("HF_TOKEN", "HF_BACKUP_REPO", "BACKUP_ENCRYPTION_KEY"):
        assert f"require_env {name}" in source
