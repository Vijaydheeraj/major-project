from flask import Flask, render_template, jsonify, Response
from src.detection.objet_detection import process_videos
from src.config.config_loader import load_config, get_video_path


def init_routes(app: Flask) -> None:
    """
    Initialize the routes for the Flask application.

    Args:
        app (Flask): The Flask application instance.

    Returns:
        None
    """

    @app.route('/')
    def index() -> str:
        """
        Render the index page.

        Returns:
            str: The rendered HTML of the index page.
        """
        return render_template('index.html')

    @app.route('/start-analysis', methods=['POST'])
    def start_analysis() -> Response:
        """
        Start the video analysis process.

        This route loads the configuration, retrieves the video path,
        and starts the video processing.

        Returns:
            Response: A JSON response indicating the status of the analysis.
        """
        config = load_config()
        config_path = get_video_path(config)
        process_videos(config_path)
        return jsonify({"status": "success", "message": "Analyse lancée!"})
