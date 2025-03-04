import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.app import create_app
from src.config.config_loader import load_config, get_video_path
from src.detection.objet_detection import process_videos

app = create_app()


if __name__ == "__main__":
    config = load_config()
    video_path = get_video_path(config)
    process_videos(video_path)
    #app.run(debug=True)
