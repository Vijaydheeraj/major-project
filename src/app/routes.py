from flask import Flask, render_template, jsonify, Response
from src.detection.objet_detection import process_videos
from src.config.config_loader import load_config, get_video_path

def init_routes(app: Flask) -> None:
    @app.route('/')
    def index() -> str:
        return render_template('index.html')

    @app.route('/start-analysis', methods=['POST'])
    def start_analysis() -> Response:
        config = load_config()
        config_path = get_video_path(config)
        process_videos(config_path)
        return jsonify({"status": "success", "message": "Analyse lancée!"})