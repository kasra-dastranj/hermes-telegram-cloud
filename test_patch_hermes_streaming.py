import pytest

from patch_hermes_streaming import patch_source


def test_patch_adds_environment_switch_and_keeps_valid_python():
    source = (
        "import os\n"
        "def run():\n"
        "                _use_streaming = True\n"
        "                return _use_streaming\n"
    )

    patched = patch_source(source)

    assert "HERMES_UPSTREAM_STREAMING" in patched
    compile(patched, "<patched>", "exec")


def test_patch_is_idempotent():
    source = (
        "import os\n"
        "def run():\n"
        "                _use_streaming = True\n"
        "                return _use_streaming\n"
    )
    patched = patch_source(source)

    assert patch_source(patched) == patched


def test_patch_fails_if_upstream_target_changes():
    with pytest.raises(RuntimeError, match="selection changed"):
        patch_source("import os\n")
