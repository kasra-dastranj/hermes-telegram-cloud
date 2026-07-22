from __future__ import annotations

import sqlite3
import tarfile
from pathlib import Path

import pytest

import backup_sync


def seed_home(home: Path) -> None:
    home.mkdir()
    with sqlite3.connect(home / "state.db") as database:
        database.execute("CREATE TABLE messages (body TEXT NOT NULL)")
        database.execute("INSERT INTO messages VALUES ('سلام')")

    (home / "sessions").mkdir()
    (home / "sessions" / "sessions.json").write_text("{}", encoding="utf-8")
    (home / "memories").mkdir()
    (home / "memories" / "USER.md").write_text("Kasra", encoding="utf-8")
    (home / "workspace").mkdir()
    (home / "workspace" / "project.txt").write_text("project", encoding="utf-8")
    (home / "workspace" / ".env").write_text("SECRET=bad", encoding="utf-8")
    (home / "workspace" / "private.pem").write_text("secret", encoding="utf-8")
    (home / "9router").mkdir()
    (home / "9router" / "data.sqlite").write_text("provider keys", encoding="utf-8")
    (home / "config.yaml").write_text("api_key: secret", encoding="utf-8")


def test_archive_is_allowlisted_and_sqlite_is_consistent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    seed_home(home)
    archive_path = tmp_path / "state.tar.gz"

    manifest = backup_sync.build_archive(home, archive_path)

    assert "state.db" in manifest["included"]
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
        assert "state.db" in names
        assert "workspace/project.txt" in names
        assert "workspace/.env" not in names
        assert "workspace/private.pem" not in names
        assert not any(name.startswith("9router") for name in names)
        assert "config.yaml" not in names
        archive.extract("state.db", path=tmp_path, filter="data")

    with sqlite3.connect(tmp_path / "state.db") as database:
        assert database.execute("SELECT body FROM messages").fetchone() == ("سلام",)


def test_streaming_encryption_round_trip_and_wrong_key(tmp_path: Path) -> None:
    plain = tmp_path / "plain.bin"
    encrypted = tmp_path / "encrypted.hbk"
    restored = tmp_path / "restored.bin"
    plain.write_bytes((b"Hermes-state-" * 100_000) + b"done")

    backup_sync.encrypt_file(plain, encrypted, "a sufficiently long backup passphrase")
    backup_sync.decrypt_file(
        encrypted, restored, "a sufficiently long backup passphrase"
    )

    assert restored.read_bytes() == plain.read_bytes()
    with pytest.raises(ValueError, match="authentication failed"):
        backup_sync.decrypt_file(
            encrypted, tmp_path / "wrong.bin", "a different and sufficiently long passphrase"
        )


def test_safe_members_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar"
    payload = tmp_path / "payload"
    payload.write_text("bad", encoding="utf-8")
    with tarfile.open(archive_path, "w") as archive:
        archive.add(payload, arcname="../outside")

    with tarfile.open(archive_path, "r") as archive:
        with pytest.raises(ValueError, match="unsafe archive member"):
            backup_sync.safe_members(archive)
