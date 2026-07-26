#!/usr/bin/env python3
"""Allow cloud deployments to disable upstream LLM streaming.

Hermes deliberately prefers streaming API calls even when platform streaming
is disabled.  That is normally useful, but a streaming response that is
aborted after HTTP headers have been sent cannot be retried by 9Router's
fallback combo.  This narrow build-time patch adds an environment switch while
preserving Hermes' default behavior everywhere else.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def resolve_target() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()

    spec = importlib.util.find_spec("agent.conversation_loop")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate agent.conversation_loop")
    return Path(spec.origin).resolve()


def patch_source(source: str) -> str:
    marker = "HERMES_UPSTREAM_STREAMING"
    if marker in source:
        return source

    needle = "                _use_streaming = True\n"
    if source.count(needle) != 1:
        raise RuntimeError(
            "Hermes' upstream streaming selection changed; "
            "refusing to apply an unsafe patch"
        )

    replacement = (
        "                _use_streaming = os.getenv(\n"
        '                    "HERMES_UPSTREAM_STREAMING", "true"\n'
        '                ).strip().lower() not in {"0", "false", "no", "off"}\n'
    )
    return source.replace(needle, replacement, 1)


def main() -> None:
    target = resolve_target()
    source = target.read_text(encoding="utf-8")
    patched = patch_source(source)
    compile(patched, str(target), "exec")

    if patched == source:
        print(f"[build] Hermes upstream streaming patch already present: {target}")
        return

    target.write_text(patched, encoding="utf-8", newline="\n")
    print(
        "[build] Patched Hermes to honor HERMES_UPSTREAM_STREAMING: "
        f"{target}"
    )


if __name__ == "__main__":
    main()
