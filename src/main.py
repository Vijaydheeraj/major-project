import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.config.config_loader import load_config, get_video_path
from src.detection.objet_detection import process_videos


def main():
    """
    Main function to load configuration, get video path, and process videos.

    This function loads the configuration, retrieves the video path from the configuration,
    and processes the videos in the specified path.
    """
    config = load_config()
    video_path = get_video_path(config)

    process_videos(video_path, 100)

if __name__ == "__main__":
    print("START.")
    main()
    print("END.")