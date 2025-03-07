import unittest
from flask.testing import FlaskClient
from src.app import create_app

class TestRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.app: FlaskClient = create_app().test_client()

    def test_index(self) -> None:
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()