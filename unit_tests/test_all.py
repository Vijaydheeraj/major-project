import unittest
from test_config import *
from test_detection import *
from test_routes import *

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="unit_tests", pattern="test_*.py")

    runner = unittest.TextTestRunner()
    runner.run(suite)