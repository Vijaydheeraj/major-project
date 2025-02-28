from flask import Flask


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    This function initializes the Flask application, sets up the application context,
    and initializes the routes.

    Returns:
        Flask: The configured Flask application instance.
    """
    app = Flask(__name__)

    with app.app_context():
        from .routes import init_routes
        init_routes(app)

    return app
