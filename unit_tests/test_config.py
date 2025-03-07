import unittest
from src.config.config_loader import load_config, get_video_path

class TestConfigLoader(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test case.

        This method loads the configuration before each test.
        """
        self.config = load_config()

    def test_load_config(self) -> None:
        """
        Test the load_config function.

        This test verifies that the configuration contains the 'videos' key.
        """
        self.assertIn('videos', self.config)

    def test_get_video_path(self) -> None:
        """
        Test the get_video_path function.

        This test verifies that the video path returned by get_video_path is a string.
        """
        video_path = get_video_path(self.config)
        self.assertIsInstance(video_path, str)

if __name__ == '__main__':
    unittest.main()