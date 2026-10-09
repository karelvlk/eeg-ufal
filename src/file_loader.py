import os
import re
import io
import logging
import json
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
from typing import Literal
from enum import Enum

LOCAL_DEBUG = True


class LoaderType(Enum):
    GITHUB = "github"
    LOCAL = "local"


class FileLoader:
    loader_type: LoaderType = LoaderType.LOCAL

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.repo_structure = None
        self.categories: list[Literal["Read", "See", "Update", "Translate", "All"]] = [
            "Read",
            "See",
            "Update",
            "Translate",
            "All",
        ]
        self.repo_structure = self.get_repo_structure()

        # Pre-index files by category and basename
        self.eeg_files: dict[str, dict[str, dict]] = {cat: {} for cat in self.categories}
        self.gaze_files: dict[str, dict[str, str]] = {cat: {} for cat in self.categories}
        self.audio_files: dict[str, dict[str, str]] = {cat: {} for cat in self.categories}
        self.index_files()

    def get_categories(self) -> list[Literal["Read", "See", "Update", "Translate", "All"]]:
        return self.categories

    def index_files(self):
        """Pre-index all data files by category and basename for faster lookups."""
        DATA_PATH = "preprocessed-data"
        EEG_PATH = "eeg"
        GAZE_PATH = "gaze"
        AUDIO_PATH = "audio"

        cat_counter = {cat: 0 for cat in self.categories}

        for item in self.repo_structure:
            if (
                item["type"] != "blob"
                or DATA_PATH not in item["path"]
                or not (item["path"].endswith(".csv") or item["path"].endswith(".wav"))
            ):
                continue

            path = item["path"]
            basename = os.path.basename(path)[:-4]

            # Get category
            category = None
            for cat in self.categories:
                if cat in path:
                    category = cat
                    break

            if category is None:
                continue

            # Categorize by file type
            if EEG_PATH in path:
                # Extract IDs
                participant_id = self.extract_participant_id(basename)
                sentence_id = self.extract_sentence_id(basename)

                self.eeg_files[category][basename] = {
                    "path": path,
                    "participant_id": participant_id,
                    "sentence_id": sentence_id,
                    "category": category,
                }

                cat_counter[category] += 1

            elif GAZE_PATH in path:
                self.gaze_files[category][basename] = path
            elif AUDIO_PATH in path:
                self.audio_files[category][basename] = path

        print("Indexing complete:", cat_counter)

    def get_repo_structure(self) -> list[dict]:
        from src.dataset_cache import enabled, index
        if enabled():
            return list(index(str(self.base_path))[1].values())
        tree = []
        for base, dirs, files in os.walk(self.base_path):
            rel_base = os.path.relpath(base, self.base_path)
            if rel_base == ".":
                rel_base = ""

            # Add directories
            for d in dirs:
                dir_path = os.path.join(rel_base, d)
                tree.append(
                    {
                        "path": dir_path.replace("\\", "/"),
                        "mode": "040000",  # mode for directories
                        "type": "tree",
                    }
                )

            # Add files
            for f in files:
                file_path = os.path.join(rel_base, f)
                tree.append(
                    {
                        "path": file_path.replace("\\", "/"),
                        "mode": "100644",  # mode for regular files
                        "type": "blob",
                    }
                )
        return tree

    def extract_participant_id(self, filename: str) -> str | None:
        """Extract participant ID (P<number>) from filename."""
        match = re.search(r"P\d+", filename)
        return match.group(0) if match else None

    def extract_sentence_id(self, filename: str) -> str | None:
        """Extract sentence ID (S<number>) from filename."""
        match = re.search(r"S\d+", filename)
        return match.group(0) if match else None

    def get_data_files(
        self,
        category_filter: Literal["Read", "See", "Update", "Translate", "All"] = "All",
        participant_id_filter: str | None = None,
        sentence_id_filter: str | None = None,
    ):
        """
        Get data files that match the participant_id and sentence_id filters.

        Args:
            category_filter: Filter by task category ("Read", "See", "Update", "Translate" or "All")
            participant_id_filter: Optional filter for participant ID (format: P<number>)
            sentence_id_filter: Optional filter for sentence ID (format: S<number>)

        Returns:
            List of data units with matching files
        """
        # Build data units using the pre-indexed files
        data_units = []

        # Determine which categories to include
        categories_to_search = (
            [category_filter] if category_filter != "All" else [cat for cat in self.categories if cat != "All"]
        )

        for category in categories_to_search:
            # Gaze-only trials are real recordings too; do not require EEG to browse them.
            basenames = sorted(set(self.eeg_files[category]) | set(self.gaze_files[category]))
            for basename in basenames:
                eeg_info = self.eeg_files[category].get(basename, {
                    "participant_id": self.extract_participant_id(basename),
                    "sentence_id": self.extract_sentence_id(basename),
                })
                # Apply filters
                if participant_id_filter and eeg_info["participant_id"] != participant_id_filter:
                    continue
                if sentence_id_filter and eeg_info["sentence_id"] != sentence_id_filter:
                    continue

                # Retrieve corresponding files
                gaze_path = self.gaze_files[category].get(basename)
                audio_path = self.audio_files[category].get(basename)

                data_unit = {
                    "name": basename,
                    "participant_id": eeg_info["participant_id"],
                    "sentence_id": eeg_info["sentence_id"],
                    "category": category,
                    "eeg": os.path.join(
                        self.base_path, "contents" if self.loader_type == LoaderType.GITHUB else "", eeg_info["path"]
                    ) if eeg_info.get("path") else None,
                    "gaze": os.path.join(
                        self.base_path, "contents" if self.loader_type == LoaderType.GITHUB else "", gaze_path
                    )
                    if gaze_path
                    else None,
                    "audio": os.path.join(
                        self.base_path, "contents" if self.loader_type == LoaderType.GITHUB else "", audio_path
                    )
                    if audio_path
                    else None,
                }

                data_units.append(data_unit)

        return data_units

    def load_data(self, data_unit: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None, io.BytesIO | None]:
        from src.dataset_cache import enabled, resolve_ui
        if enabled():
            data_unit = dict(data_unit)
            for kind in ("eeg", "gaze", "audio"):
                if data_unit.get(kind):
                    relative = str(Path(data_unit[kind]).relative_to(Path(self.base_path)))
                    resolved = resolve_ui(self.base_path, relative)
                    data_unit[kind] = str(resolved) if resolved else None
        eeg = None
        gaze = None
        audio_bytes = None

        def read_signal(path, kind):
            if not path:
                return None
            try:
                frame = pd.read_csv(path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                st.warning(f"{kind} CSV could not be parsed: {Path(path).name}. {exc}")
                return None
            if frame.empty:
                st.warning(
                    f"The published {kind} CSV contains headers but no samples: {Path(path).name}. "
                    "This is missing recording data, not absence of activity."
                )
                return None
            return frame

        eeg = read_signal(data_unit["eeg"], "EEG")
        gaze = read_signal(data_unit["gaze"], "gaze")
        if data_unit["audio"]:
            with open(data_unit["audio"], "rb") as f:
                audio_bytes = io.BytesIO(f.read())
            audio_bytes.seek(0)

        return eeg, gaze, audio_bytes

    def get_valid_participant_ids(
        self,
        category_filter: Literal["Read", "See", "Update", "Translate", "All"] = "All",
        sentence_id_filter: str | None = None,
    ) -> list[str]:
        data_units = self.get_data_files(category_filter, sentence_id_filter=sentence_id_filter)
        return sorted({du["participant_id"] for du in data_units if du["participant_id"]})

    def get_valid_sentence_ids(
        self,
        category_filter: Literal["Read", "See", "Update", "Translate", "All"] = "All",
        participant_id_filter: str | None = None,
    ) -> list[str]:
        data_units = self.get_data_files(category_filter, participant_id_filter)
        return sorted({du["sentence_id"] for du in data_units if du["sentence_id"]})


class GithubFileLoader(FileLoader):
    loader_type: LoaderType = LoaderType.GITHUB

    def __init__(self, base_url: str):
        # Convert GitHub API URL to raw content URL format if needed
        if "api.github.com/repos" in base_url:
            # Extract owner and repo from API URL
            parts = base_url.split("repos/")[1].split("/")
            owner, repo = parts[0], parts[1]
            self.raw_base_url = f"https://raw.githubusercontent.com/{owner}/{repo}/refs/heads/main"
        else:
            self.raw_base_url = base_url

        self.api_base_url = base_url
        self.index_warning = None
        cache_root = Path(os.getenv("GITHUB_INDEX_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache")))
        repository_key = re.sub(r"[^a-zA-Z0-9_-]", "_", base_url.split("repos/")[-1].strip("/"))
        self.index_cache_path = cache_root / f"github-index-{repository_key}.json"
        super().__init__(base_path=base_url)

    def get_repo_structure(self):
        """Retry transient failures; use only a complete, same-repository cached index."""
        url = f"{self.api_base_url}/git/trees/main?recursive=1"
        error = "unknown error"
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                payload = response.json()
                if not self._valid_index(payload):
                    raise ValueError("GitHub returned an incomplete or invalid dataset index")
                try:
                    self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.index_cache_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload))
                    temporary.replace(self.index_cache_path)
                except OSError as exc:
                    logging.warning("Dataset index could not be cached: %s", exc)
                return payload["tree"]
            except (requests.RequestException, ValueError) as exc:
                error = str(exc)
                logging.warning("Dataset index attempt %s failed: %s", attempt + 1, error)
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    if exc.response.status_code < 500 and exc.response.status_code != 429:
                        break
        try:
            payload = json.loads(self.index_cache_path.read_text())
            if not self._valid_index(payload):
                raise ValueError("Invalid cached index")
            # Cached paths must fetch files from the SAME commit, not today's main.
            self.raw_base_url = self.raw_base_url.rsplit("/", 1)[0].removesuffix("/refs/heads") + "/" + payload["sha"]
            self.index_warning = f"GitHub index is unavailable. Using saved index at commit {payload['sha'][:12]}; file downloads still need network access."
            return payload["tree"]
        except (OSError, ValueError, KeyError):
            raise RuntimeError("GitHub could not provide the dataset index and no valid saved index is available. Retry, or choose a populated local dataset folder. Details: " + error) from None

    def _valid_index(self, payload):
        return (isinstance(payload, dict) and isinstance(payload.get("tree"), list)
                and bool(payload["tree"]) and not payload.get("truncated", False)
                and bool(re.fullmatch(r"[0-9a-f]{40}", str(payload.get("sha", ""))))
                and str(payload.get("url", "")).startswith(self.api_base_url.rstrip("/") + "/git/trees/"))

    def get_data_files(
        _self,
        category_filter: Literal["Read", "See", "Update", "Translate", "All"] = "All",
        participant_id_filter: str | None = None,
        sentence_id_filter: str | None = None,
    ):
        """
        Get data files that match the participant_id and sentence_id filters.

        Args:
            category_filter: Filter by task category ("Read", "See", "Update", "Translate" or "All")
            participant_id_filter: Optional filter for participant ID (format: P<number>)
            sentence_id_filter: Optional filter for sentence ID (format: S<number>)

        Returns:
            List of data units with matching files
        """
        # Build data units using the pre-indexed files
        data_units = []

        # Determine which categories to include
        categories_to_search = (
            [category_filter] if category_filter != "All" else [cat for cat in _self.categories if cat != "All"]
        )

        for category in categories_to_search:
            for basename in sorted(set(_self.eeg_files[category]) | set(_self.gaze_files[category])):
                eeg_info = _self.eeg_files[category].get(basename, {
                    "participant_id": _self.extract_participant_id(basename),
                    "sentence_id": _self.extract_sentence_id(basename),
                })
                # Apply filters
                if participant_id_filter and eeg_info["participant_id"] != participant_id_filter:
                    continue
                if sentence_id_filter and eeg_info["sentence_id"] != sentence_id_filter:
                    continue

                # Retrieve corresponding files
                gaze_path = _self.gaze_files[category].get(basename)
                audio_path = _self.audio_files[category].get(basename)

                data_unit = {
                    "name": basename,
                    "participant_id": eeg_info["participant_id"],
                    "sentence_id": eeg_info["sentence_id"],
                    "category": category,
                    "eeg": f"{_self.raw_base_url}/{eeg_info['path']}" if eeg_info.get("path") else None,
                    "gaze": f"{_self.raw_base_url}/{gaze_path}" if gaze_path else None,
                    "audio": f"{_self.raw_base_url}/{audio_path}" if audio_path else None,
                }

                data_units.append(data_unit)

        return data_units

    @st.cache_data
    def load_data(_self, data_unit: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None, io.BytesIO | None]:
        """
        Load data from GitHub, with Streamlit caching.

        Returns:
            Tuple of (eeg, gaze, audio) data
        """
        print("calling apis")
        print("   eeg:", data_unit["eeg"])
        print("   gaze:", data_unit["gaze"])
        print("   audio:", data_unit["audio"])

        eeg = None
        gaze = None
        audio = None

        if data_unit["eeg"]:
            try:
                eeg = pd.read_csv(data_unit["eeg"])
            except Exception as e:
                if LOCAL_DEBUG:
                    logging.error(f"Failed to load EEG data: {e}")
                else:
                    st.error(f"Failed to load EEG data: {e}")

        if data_unit["gaze"]:
            try:
                gaze = pd.read_csv(data_unit["gaze"])
            except Exception as e:
                if LOCAL_DEBUG:
                    logging.error(f"Failed to load gaze data: {e}")
                else:
                    st.error(f"Failed to load gaze data: {e}")

        if data_unit["audio"]:
            try:
                response = requests.get(data_unit["audio"], timeout=30)
                if response.status_code == 200:
                    audio = io.BytesIO(response.content)
            except Exception as e:
                if LOCAL_DEBUG:
                    logging.error(f"Failed to load audio data: {e}")
                else:
                    st.error(f"Failed to load audio data: {e}")

        return eeg, gaze, audio


if __name__ == "__main__":
    loader = GithubFileLoader(base_url="https://api.github.com/repos/ufal/eyetracked-multi-modal-translation")
    # loader = FileLoader(base_path="../ufal_emmt")
    ds = loader.get_data_files(category_filter="Read", participant_id_filter="P03", sentence_id_filter="S084")
    # ds = loader.get_data_files(category_filter="Read", participant_id_filter="P03", sentence_id_filter="S092")
    for d in ds:
        print("\n>", d)
    du = loader.get_data_files(category_filter="Read", participant_id_filter=None, sentence_id_filter=None)
    # # print(du)
    # eeg, gaze, audio = loader.load_data(du)
    # print(eeg)
    # print(gaze)
