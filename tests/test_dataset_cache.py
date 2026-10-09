import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from src.dataset_cache import available, resolve, blob_sha, index, safe_path

class DatasetCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.body = b"timestamp,X,Y\n1,10,20\n"
        self.relative = "preprocessed-data/gaze/See/P01.csv"
        tree = {"sha": "a" * 40, "truncated": False, "tree": [{"path": self.relative, "type": "blob", "size": len(self.body), "sha": blob_sha(self.body)}]}
        (self.root / "repository_tree.json").write_text(json.dumps(tree))
        self.env = patch.dict(os.environ, {"EMMT_LAZY_DOWNLOAD": "1"})
        self.env.start()
    def tearDown(self):
        self.env.stop()
        index.cache_clear()
        self.directory.cleanup()
    def test_availability_does_not_download(self):
        with patch("src.dataset_cache.requests.get") as get:
            self.assertTrue(available(self.root, self.relative))
            self.assertFalse(available(self.root, "missing.csv"))
            get.assert_not_called()
    def test_verified_download_then_offline_cache(self):
        response = Mock(content=self.body)
        with patch("src.dataset_cache.requests.get", return_value=response) as get:
            path = resolve(self.root, self.relative)
            self.assertIn("/" + "a" * 40 + "/", get.call_args.args[0])
            self.assertEqual(path.read_bytes(), self.body)
            resolve(self.root, self.relative)
            self.assertEqual(get.call_count, 1)
    def test_corrupt_response_never_saved(self):
        with patch("src.dataset_cache.requests.get", return_value=Mock(content=b"bad")):
            with self.assertRaises(RuntimeError):
                resolve(self.root, self.relative)
        self.assertFalse((self.root / self.relative).exists())
    def test_paths_cannot_escape_cache(self):
        for relative in ("../outside", "/outside", "a\\b"):
            with self.assertRaises(ValueError):
                safe_path(self.root, relative)
    def test_missing_file_does_not_fetch_another(self):
        with patch("src.dataset_cache.requests.get") as get:
            with self.assertRaises(FileNotFoundError):
                resolve(self.root, "missing.csv")
            get.assert_not_called()

if __name__ == "__main__":
    unittest.main()
