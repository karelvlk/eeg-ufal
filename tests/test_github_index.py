import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import requests
from src.file_loader import GithubFileLoader

BASE = "https://api.github.com/repos/ufal/eyetracked-multi-modal-translation"
SHA = "a" * 40
PAYLOAD = {"sha": SHA, "url": BASE + "/git/trees/" + SHA,
           "tree": [{"type": "blob", "path": "preprocessed-data/eeg/Read/P01_S001_Read.csv"}]}


class IndexTests(unittest.TestCase):
    def loader(self, folder):
        loader = GithubFileLoader.__new__(GithubFileLoader)
        loader.api_base_url = BASE
        loader.raw_base_url = "https://raw.githubusercontent.com/ufal/eyetracked-multi-modal-translation/refs/heads/main"
        loader.index_cache_path = Path(folder) / "index.json"
        loader.index_warning = None
        return loader

    def test_500_uses_saved_same_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            loader = self.loader(folder)
            loader.index_cache_path.write_text(json.dumps(PAYLOAD))
            response = Mock(status_code=500)
            response.raise_for_status.side_effect = requests.HTTPError("500", response=response)
            with patch("src.file_loader.requests.get", return_value=response) as get:
                self.assertEqual(loader.get_repo_structure(), PAYLOAD["tree"])
                self.assertEqual(get.call_count, 3)
            self.assertTrue(loader.raw_base_url.endswith("/" + SHA))
            self.assertNotIn("refs/heads", loader.raw_base_url)
            self.assertIsNotNone(loader.index_warning)

    def test_success_saves_index(self):
        with tempfile.TemporaryDirectory() as folder:
            loader = self.loader(folder)
            response = Mock()
            response.json.return_value = PAYLOAD
            with patch("src.file_loader.requests.get", return_value=response):
                self.assertEqual(loader.get_repo_structure(), PAYLOAD["tree"])
            self.assertEqual(json.loads(loader.index_cache_path.read_text()), PAYLOAD)

    def test_file_lists_use_each_loaders_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict("os.environ", {"GITHUB_INDEX_CACHE_DIR": folder}):
                with patch.object(GithubFileLoader, "get_repo_structure", return_value=PAYLOAD["tree"]):
                    first = GithubFileLoader(BASE)
                    second = GithubFileLoader(BASE)
            first.raw_base_url = "https://raw.githubusercontent.com/ufal/eyetracked-multi-modal-translation/" + SHA
            second.raw_base_url = "https://raw.githubusercontent.com/ufal/eyetracked-multi-modal-translation/" + "b" * 40
            self.assertIn(SHA, first.get_data_files()[0]["eeg"])
            self.assertIn("b" * 40, second.get_data_files()[0]["eeg"])

    def test_timeout_without_cache_is_controlled_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("src.file_loader.requests.get", side_effect=requests.Timeout("timeout")):
                with self.assertRaisesRegex(RuntimeError, "no valid saved index"):
                    self.loader(folder).get_repo_structure()

    def test_wrong_repository_or_truncated_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            loader = self.loader(folder)
            self.assertFalse(loader._valid_index({**PAYLOAD, "url": "https://example.com/git/trees/" + SHA}))
            self.assertFalse(loader._valid_index({**PAYLOAD, "truncated": True}))


if __name__ == "__main__":
    unittest.main()
