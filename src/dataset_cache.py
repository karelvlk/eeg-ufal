"""Commit-pinned downloads into a verified local cache."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from functools import lru_cache
import requests

def enabled():
    return os.getenv("EMMT_LAZY_DOWNLOAD", "0") == "1"

@lru_cache(maxsize=4)
def index(root):
    payload = json.loads((Path(root) / "repository_tree.json").read_text())
    sha = str(payload.get("sha", ""))
    if payload.get("truncated") or not payload.get("tree") or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("A complete, commit-pinned dataset index is required.")
    return sha, {i["path"]: i for i in payload["tree"] if i.get("type") == "blob"}

def safe_path(root, relative):
    path = PurePosixPath(str(relative))
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in str(relative):
        raise ValueError("Invalid relative dataset path")
    target = Path(root) / str(relative)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset path escapes cache")
    return target

def available(root, relative):
    if not relative:
        return False
    target = safe_path(root, relative)
    return str(relative) in index(str(root))[1] if enabled() else target.is_file()

def blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()

def resolve(root, relative):
    target = safe_path(root, relative)
    if not enabled():
        if not target.is_file():
            raise FileNotFoundError(str(target))
        return target
    commit, files = index(str(root))
    entry = files.get(str(relative))
    if entry is None:
        raise FileNotFoundError(f"Not in pinned dataset: {relative}")
    if target.is_file() and blob_sha(target.read_bytes()) == entry["sha"]:
        return target
    url = f"https://raw.githubusercontent.com/ufal/eyetracked-multi-modal-translation/{commit}/{relative}"
    error = None
    for _ in range(3):
        temporary = None
        try:
            response = requests.get(url, timeout=(5, 45))
            response.raise_for_status()
            content = response.content
            if len(content) != entry.get("size") or blob_sha(content) != entry["sha"]:
                raise ValueError(f"Dataset checksum mismatch: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
            temporary.replace(target)
            return target
        except (requests.RequestException, OSError, ValueError) as exc:
            error = exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not load {relative}: {error}")

def resolve_ui(root, relative):
    if not relative:
        return None
    import streamlit as st
    try:
        path = resolve(root, relative)
        if path.stat().st_size == 0:
            st.warning(f"The published file is empty: {relative}. This is missing data, not absence of activity.")
            return None
        return path
    except (RuntimeError, ValueError, OSError) as exc:
        st.warning(str(exc))
        return None
