#!/usr/bin/env python3
"""Force Hermes' built-in Groq transcription handler to send a language.

Hermes currently calls Groq's transcription endpoint without forwarding a
language. Short Persian voice notes can therefore be auto-detected poorly and
occasionally produce English text. This build-time patch is intentionally
narrow and fails loudly if the upstream function changes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def resolve_target() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()

    spec = importlib.util.find_spec("tools.transcription_tools")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate tools.transcription_tools")
    return Path(spec.origin).resolve()


def patch_source(source: str) -> str:
    start = source.find("def _transcribe_groq(")
    end = source.find("\ndef _transcribe_openai(", start)
    if start < 0 or end < 0:
        raise RuntimeError("could not isolate Hermes' Groq STT handler")

    block = source[start:end]
    if "STT_GROQ_LANGUAGE" in block:
        return source

    needle = '                    response_format="text",\n'
    if block.count(needle) != 1:
        raise RuntimeError(
            "Hermes' Groq STT call changed; refusing to apply an unsafe patch"
        )

    replacement = (
        needle
        + '                    language=os.getenv("STT_GROQ_LANGUAGE", "fa").strip() or "fa",\n'
    )
    patched_block = block.replace(needle, replacement, 1)
    return source[:start] + patched_block + source[end:]


def main() -> None:
    target = resolve_target()
    source = target.read_text(encoding="utf-8")
    patched = patch_source(source)
    compile(patched, str(target), "exec")

    if patched == source:
        print(f"[build] Hermes Groq STT language patch already present: {target}")
        return

    target.write_text(patched, encoding="utf-8", newline="\n")
    print(f"[build] Patched Hermes Groq STT to honor STT_GROQ_LANGUAGE: {target}")


if __name__ == "__main__":
    main()
