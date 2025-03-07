import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Functions to unit_tests
from src.detection.objet_detection import process_frame, process_video, process_videos


@patch('src.detection.objet_detection.detection_yolov11')
@patch('src.detection.objet_detection.detection_yolov11_fine_tuning')
@patch('src.detection.objet_detection.classification_fine_tuning')
class TestDetection(unittest.TestCase):
    def test_process_frame(self, mock_classification, mock_fine_tuning, mock_detection):
        """
        Test the process_frame function.

        This test verifies that the process_frame function correctly processes a frame
        and returns the expected detection and classification results.

        Args:
            mock_classification (MagicMock): Mock for the classification function.
            mock_fine_tuning (MagicMock): Mock for the fine-tuning detection function.
            mock_detection (MagicMock): Mock for the detection function.
        """
        # Create a mock frame as a NumPy array
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_detection.return_value = pd.DataFrame({'xmin': [0], 'ymin': [0], 'xmax': [100], 'ymax': [100], 'name': ['object'], 'confidence': [0.9]})
        mock_fine_tuning.return_value = pd.DataFrame({'xmin': [10], 'ymin': [10], 'xmax': [90], 'ymax': [90], 'name': ['object'], 'confidence': [0.95]})
        mock_classification.return_value = [MagicMock(class_name='empty', confidence=0.85)]

        detections_df, classification_df_finetuning = process_frame(mock_frame)

        mock_detection.assert_called_once_with(mock_frame)
        mock_fine_tuning.assert_called_once_with(mock_frame)
        mock_classification.assert_called_once_with(mock_frame)

        self.assertIsInstance(detections_df, pd.DataFrame)
        self.assertIsInstance(classification_df_finetuning, list)
        self.assertEqual(len(classification_df_finetuning), 1)
        self.assertEqual(classification_df_finetuning[0].class_name, 'empty')


@patch('cv2.VideoCapture')
@patch('cv2.namedWindow')
@patch('cv2.resizeWindow')
@patch('cv2.imshow')
@patch('cv2.waitKey')
@patch('cv2.destroyAllWindows')
@patch('src.detection.objet_detection.detection_yolov11')
@patch('src.detection.objet_detection.detection_yolov11_fine_tuning')
@patch('src.detection.objet_detection.classification_fine_tuning')
class TestVideoProcessing(unittest.TestCase):
    def test_process_video(self, mock_classification, mock_fine_tuning, mock_detection, mock_destroy, mock_wait, mock_imshow, mock_resize, mock_named, mock_capture):
        """
        Test the process_video function.

        This test verifies that the process_video function correctly processes a video
        and calls the expected detection and classification functions.

        Args:
            mock_classification (MagicMock): Mock for the classification function.
            mock_fine_tuning (MagicMock): Mock for the fine-tuning detection function.
            mock_detection (MagicMock): Mock for the detection function.
            mock_destroy (MagicMock): Mock for the cv2.destroyAllWindows function.
            mock_wait (MagicMock): Mock for the cv2.waitKey function.
            mock_imshow (MagicMock): Mock for the cv2.imshow function.
            mock_resize (MagicMock): Mock for the cv2.resizeWindow function.
            mock_named (MagicMock): Mock for the cv2.namedWindow function.
            mock_capture (MagicMock): Mock for the cv2.VideoCapture function.
        """
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, mock_frame), (True, mock_frame), (False, None)]
        mock_capture.return_value = mock_cap

        mock_detection.return_value = pd.DataFrame({'xmin': [0], 'ymin': [0], 'xmax': [100], 'ymax': [100], 'name': ['object'], 'confidence': [0.9]})
        mock_fine_tuning.return_value = pd.DataFrame({'xmin': [10], 'ymin': [10], 'xmax': [90], 'ymax': [90], 'name': ['object'], 'confidence': [0.95]})
        mock_classification.return_value = [MagicMock(class_name='empty', confidence=0.85)]

        process_video('fake_video.mp4', nb_of_img_skip_between_2=0)

        mock_capture.assert_called_once_with('fake_video.mp4')
        self.assertEqual(mock_cap.read.call_count, 3)
        mock_detection.assert_called()
        mock_fine_tuning.assert_called()
        mock_classification.assert_called()
        mock_destroy.assert_called_once()


@patch('os.listdir')
@patch('src.detection.objet_detection.process_video')
class TestVideosProcessing(unittest.TestCase):
    def test_process_videos(self, mock_process_video, mock_listdir):
        """
        Test the process_videos function.

        This test verifies that the process_videos function correctly processes all videos
        in a specified directory and calls the process_video function for each video.

        Args:
            mock_process_video (MagicMock): Mock for the process_video function.
            mock_listdir (MagicMock): Mock for the os.listdir function.
        """
        mock_listdir.return_value = ['video1.mp4', 'video2.mp4']  # Simulate a folder with 2 video files

        process_videos('fake_folder', nb_of_img_skip_between_2=0)

        mock_listdir.assert_called_once_with('fake_folder')
        self.assertEqual(mock_process_video.call_count, 2)
        mock_process_video.assert_any_call(os.path.join('fake_folder', 'video1.mp4'), 0)
        mock_process_video.assert_any_call(os.path.join('fake_folder', 'video2.mp4'), 0)

if __name__ == '__main__':
    unittest.main()