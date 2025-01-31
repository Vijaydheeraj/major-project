import unittest
from src.config.config_loader import load_config, get_video_path

class TestConfigLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_load_config(self) -> None:
        self.assertIn('videos', self.config)

    def test_get_video_path(self) -> None:
        video_path = get_video_path(self.config)
        self.assertIsInstance(video_path, str)

if __name__ == '__main__':
    unittest.main()