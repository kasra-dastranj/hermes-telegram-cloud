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
    assert "BACKUP_ALLOW_MISSING_REMOTE=false" in dockerfile
    assert "fallback-models.json" in (ROOT / "configure_9router.js").read_text(
        encoding="utf-8"
    )
    assert "fallback-models.json" in (ROOT / "model_proxy.py").read_text(
        encoding="utf-8"
    )


def test_long_tasks_and_context_have_defensive_settings():
    config = generated_config()

    assert config["agent"]["intent_ack_continuation"] is True
    assert config["agent"]["api_max_retries"] == 1
    assert config["compression"]["in_place"] is True
    assert config["compression"]["threshold_tokens"] == 36000
    assert config["compression"]["proactive_prune_tokens"] == 24000


def test_startup_fails_closed_on_backup_restore_errors():
    source = (ROOT / "start.sh").read_text(encoding="utf-8")

    assert "backup_sync.py restore || true" not in source
    assert "if [ \"$restore_status\" -eq 1 ]" in source
    for name in ("HF_TOKEN", "HF_BACKUP_REPO", "BACKUP_ENCRYPTION_KEY"):
        assert f"require_env {name}" in source
