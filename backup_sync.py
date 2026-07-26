#!/usr/bin/env python3
"""Encrypted, secret-aware Hermes state backup for ephemeral deployments.

The backup is deliberately allow-list based. 9Router's database, Hermes
configuration, environment files, credentials, and provider tokens are never
included. The resulting archive is encrypted locally before it is uploaded to
a private Hugging Face Dataset repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MAGIC = b"HERMES-BACKUP-V1\n"
SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16
CHUNK_SIZE = 1024 * 1024

ALLOWED_DIRS = (
    "sessions",
    "memories",
    "cron",
    "workspace",
    "checkpoints",
)
ALLOWED_FILES = (
    "state.db",
    "USER.md",
    "SOUL.md",
    "MEMORY.md",
    "AGENTS.md",
    "cron-jobs.json",
)
EXCLUDED_NAMES = {
    ".env",
    "auth.json",
    "credentials.json",
    "config.yaml",
    "gateway.lock",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
}
EXCLUDED_SUFFIXES = (".key", ".pem", ".p12", ".pfx", ".token")


def log(message: str) -> None:
    print(f"[backup] {message}", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def enabled() -> bool:
    return all(
        str(os.environ.get(name, "")).strip()
        for name in ("HF_TOKEN", "HF_BACKUP_REPO", "BACKUP_ENCRYPTION_KEY")
    )


def allow_missing_remote() -> bool:
    return str(
        os.environ.get("BACKUP_ALLOW_MISSING_REMOTE", "true")
    ).strip().lower() in {"1", "true", "yes", "on"}


def settings() -> tuple[Path, str, str, str]:
    home = Path(os.environ.get("HERMES_HOME", "/opt/data")).resolve()
    repo_id = str(os.environ.get("HF_BACKUP_REPO", "")).strip()
    token = str(os.environ.get("HF_TOKEN", "")).strip()
    passphrase = str(os.environ.get("BACKUP_ENCRYPTION_KEY", ""))
    if len(passphrase) < 24:
        raise ValueError("BACKUP_ENCRYPTION_KEY must contain at least 24 characters")
    return home, repo_id, token, passphrase


def should_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        lowered = name.lower()
        if (
            name in EXCLUDED_NAMES
            or lowered.startswith(".env.")
            or lowered.startswith("credentials")
            or lowered.startswith("token")
            or lowered.endswith(EXCLUDED_SUFFIXES)
        ):
            ignored.add(name)
    return ignored


def sqlite_snapshot(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{source.as_posix()}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True, timeout=15)) as source_db:
        with closing(sqlite3.connect(destination, timeout=15)) as destination_db:
            source_db.backup(destination_db, pages=256, sleep=0.05)


def build_archive(home: Path, output: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="hermes-backup-stage-") as raw_stage:
        stage = Path(raw_stage)
        copied: list[str] = []

        state_db = home / "state.db"
        if state_db.is_file():
            sqlite_snapshot(state_db, stage / "state.db")
            copied.append("state.db")

        for relative in ALLOWED_FILES:
            if relative == "state.db":
                continue
            source = home / relative
            if source.is_file() and source.name not in EXCLUDED_NAMES:
                shutil.copy2(source, stage / relative)
                copied.append(relative)

        for relative in ALLOWED_DIRS:
            source = home / relative
            if source.is_dir():
                shutil.copytree(
                    source,
                    stage / relative,
                    dirs_exist_ok=True,
                    ignore=should_ignore,
                )
                copied.append(relative + "/")

        manifest = {
            "format": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "included": sorted(set(copied)),
        }
        (stage / "backup-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with tarfile.open(output, "w:gz", compresslevel=6) as archive:
            for item in sorted(stage.iterdir(), key=lambda path: path.name):
                archive.add(item, arcname=item.name, recursive=True)

    return manifest


def derive_key(passphrase: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    return PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    ).derive(passphrase.encode("utf-8"))


def encrypt_file(source: Path, destination: Path, passphrase: str) -> None:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    encryptor = Cipher(algorithms.AES(derive_key(passphrase, salt)), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(MAGIC)

    with source.open("rb") as source_file, destination.open("wb") as destination_file:
        destination_file.write(MAGIC)
        destination_file.write(salt)
        destination_file.write(nonce)
        while chunk := source_file.read(CHUNK_SIZE):
            destination_file.write(encryptor.update(chunk))
        destination_file.write(encryptor.finalize())
        destination_file.write(encryptor.tag)


def decrypt_file(source: Path, destination: Path, passphrase: str) -> None:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    minimum_size = len(MAGIC) + SALT_SIZE + NONCE_SIZE + TAG_SIZE
    if source.stat().st_size < minimum_size:
        raise ValueError("backup object is truncated")

    with source.open("rb") as source_file:
        if source_file.read(len(MAGIC)) != MAGIC:
            raise ValueError("unsupported backup format")
        salt = source_file.read(SALT_SIZE)
        nonce = source_file.read(NONCE_SIZE)
        ciphertext_size = source.stat().st_size - minimum_size
        source_file.seek(-TAG_SIZE, os.SEEK_END)
        tag = source_file.read(TAG_SIZE)
        source_file.seek(len(MAGIC) + SALT_SIZE + NONCE_SIZE)

        decryptor = Cipher(
            algorithms.AES(derive_key(passphrase, salt)), modes.GCM(nonce, tag)
        ).decryptor()
        decryptor.authenticate_additional_data(MAGIC)

        try:
            with destination.open("wb") as destination_file:
                remaining = ciphertext_size
                while remaining:
                    chunk = source_file.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        raise ValueError("backup ciphertext is truncated")
                    destination_file.write(decryptor.update(chunk))
                    remaining -= len(chunk)
                destination_file.write(decryptor.finalize())
        except InvalidTag as error:
            destination.unlink(missing_ok=True)
            raise ValueError("backup authentication failed; encryption key is incorrect") from error


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    allowed_roots = set(ALLOWED_DIRS) | set(ALLOWED_FILES) | {"backup-manifest.json"}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        if path.parts[0] not in allowed_roots:
            raise ValueError(f"unexpected archive member: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"unsupported archive member: {member.name}")
        members.append(member)
    return members


def upload_backup() -> bool:
    if not enabled():
        log("Remote backup is disabled; set HF_TOKEN, HF_BACKUP_REPO, and BACKUP_ENCRYPTION_KEY.")
        return False

    from huggingface_hub import CommitOperationAdd, HfApi

    home, repo_id, token, passphrase = settings()
    max_bytes = int(os.environ.get("BACKUP_MAX_BYTES", str(250 * 1024 * 1024)))
    remote_path = str(os.environ.get("HF_BACKUP_PATH", "backups/render/latest.hbk")).strip("/")

    with tempfile.TemporaryDirectory(prefix="hermes-backup-") as raw_temp:
        temp = Path(raw_temp)
        archive_path = temp / "state.tar.gz"
        encrypted_path = temp / "latest.hbk"
        manifest = build_archive(home, archive_path)
        if archive_path.stat().st_size > max_bytes:
            raise ValueError(
                f"backup archive is {archive_path.stat().st_size} bytes; limit is {max_bytes}"
            )
        encrypt_file(archive_path, encrypted_path, passphrase)

        metadata = {
            **manifest,
            "encrypted": True,
            "sha256": file_sha256(encrypted_path),
            "size": encrypted_path.stat().st_size,
        }
        metadata_path = temp / "latest.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update encrypted Hermes state backup",
            operations=[
                CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=encrypted_path),
                CommitOperationAdd(
                    path_in_repo=remote_path + ".json", path_or_fileobj=metadata_path
                ),
            ],
        )
        # Encrypted snapshots cannot benefit from binary deduplication because
        # every ciphertext is intentionally unique. Keep a small rollback
        # window, then squash history so storage cannot grow without bound.
        try:
            max_commits = max(2, int(os.environ.get("BACKUP_MAX_COMMITS", "12")))
            commits = api.list_repo_commits(repo_id=repo_id, repo_type="dataset")
            if len(commits) >= max_commits:
                api.super_squash_history(
                    repo_id=repo_id,
                    repo_type="dataset",
                    branch="main",
                    commit_message="Compact Hermes backup history",
                )
        except Exception as error:
            log(f"Warning: backup uploaded but history compaction failed: {error}")
        log(
            f"Uploaded encrypted state ({encrypted_path.stat().st_size} bytes, "
            f"{len(manifest['included'])} state groups)."
        )
        return True


def restore_backup(force: bool = False) -> bool:
    if not enabled():
        log("Remote restore is disabled; starting with local state.")
        return False

    from huggingface_hub import hf_hub_download

    home, repo_id, token, passphrase = settings()
    if not force and (home / "state.db").is_file():
        log("Local state.db already exists; remote restore skipped.")
        return False

    remote_path = str(os.environ.get("HF_BACKUP_PATH", "backups/render/latest.hbk")).strip("/")
    with tempfile.TemporaryDirectory(prefix="hermes-restore-") as raw_temp:
        temp = Path(raw_temp)
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=remote_path,
                    token=token,
                    local_dir=temp / "download",
                )
            )
        except Exception as error:
            text = str(error).lower()
            if "404" in text or "entry not found" in text or "repository not found" in text:
                if not allow_missing_remote():
                    raise RuntimeError(
                        "required remote backup is missing or inaccessible"
                    ) from error
                log("No remote backup exists yet; starting with empty state.")
                return False
            raise

        archive_path = temp / "state.tar.gz"
        decrypt_file(downloaded, archive_path, passphrase)
        home.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = safe_members(archive)
            archive.extractall(home, members=members, filter="data")
        log(f"Restored encrypted state from {repo_id}/{remote_path}.")
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("backup", "restore", "check"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        if args.action == "check":
            if not enabled():
                log("Backup configuration is incomplete.")
                return 2
            _, repo_id, _, _ = settings()
            log(f"Backup configuration is valid for private dataset {repo_id}.")
            return 0
        if args.action == "backup":
            return 0 if upload_backup() else 2
        return 0 if restore_backup(force=args.force) else 2
    except Exception as error:
        log(f"{args.action} failed: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
