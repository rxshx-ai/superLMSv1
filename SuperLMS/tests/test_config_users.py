"""Tests for config.load_users — multi-user parsing and single-user fallback."""

import json
import importlib
from pathlib import Path

import pytest

config = importlib.import_module("config")


def _write(tmp_path: Path, data) -> Path:
    f = tmp_path / "users.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    return f


def test_loads_multiple_users(tmp_path):
    f = _write(tmp_path, [
        {"name": "alice", "username": "u1", "password": "p1"},
        {"name": "bob", "username": "u2", "password": "p2"},
    ])
    users = config.load_users(users_file=f)
    assert [u["name"] for u in users] == ["alice", "bob"]
    assert users[0] == {"name": "alice", "username": "u1", "password": "p1"}


def test_name_defaults_to_username_when_absent(tmp_path):
    f = _write(tmp_path, [{"username": "u1", "password": "p1"}])
    users = config.load_users(users_file=f)
    assert users[0]["name"] == "u1"


def test_missing_password_raises(tmp_path):
    f = _write(tmp_path, [{"name": "alice", "username": "u1"}])
    with pytest.raises(ValueError, match="missing username or password"):
        config.load_users(users_file=f)


def test_non_list_raises(tmp_path):
    f = _write(tmp_path, {"username": "u1", "password": "p1"})
    with pytest.raises(ValueError, match="must be a JSON array"):
        config.load_users(users_file=f)


def test_malformed_json_raises(tmp_path):
    f = tmp_path / "users.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        config.load_users(users_file=f)


def test_fallback_to_env_single_user(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setenv("MOODLE_USERNAME", "solo")
    monkeypatch.setenv("MOODLE_PASSWORD", "secret")
    users = config.load_users(users_file=missing)
    assert users == [{"name": "solo", "username": "solo", "password": "secret"}]


def test_empty_when_nothing_configured(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.delenv("MOODLE_USERNAME", raising=False)
    monkeypatch.delenv("MOODLE_PASSWORD", raising=False)
    assert config.load_users(users_file=missing) == []
