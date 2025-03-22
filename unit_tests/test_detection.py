import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import cv2
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Functions to unit_tests
from src.detection.objet_detection import process_frame, process_video, process_videos

@patch('src.detection.background_substraction.background_sub.match_frame_reference', return_value=np.zeros((480, 640, 3), dtype=np.uint8))
@patch('src.detection.objet_detection.detection_yolov11')
@patch('src.detection.objet_detection.detection_yolov11_fine_tuning')
@patch('src.detection.objet_detection.classification_fine_tuning')
class TestDetection(unittest.TestCase):
    def test_process_frame(self, mock_classification, mock_fine_tuning, mock_detection, mock_match_frame):
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

        mock_detection.return_value = pd.DataFrame(
            {'xmin': [0], 'ymin': [0], 'xmax': [100], 'ymax': [100], 'name': ['object'], 'confidence': [0.9]})
        mock_fine_tuning.return_value = pd.DataFrame(
            {'xmin': [10], 'ymin': [10], 'xmax': [90], 'ymax': [90], 'name': ['object'], 'confidence': [0.95]})
        mock_classification.return_value = [MagicMock(class_name='empty', confidence=0.85)]
        mock_match_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)  # Image de test valide

        detections_df, detections_df_finetuning, classification_df_finetuning, detections_df_subtraction, detections_df_edgedetection = process_frame(mock_frame, 4)

        mock_detection.assert_called_once_with(mock_frame)
        mock_fine_tuning.assert_called_once_with(mock_frame)
        mock_classification.assert_called_once_with(mock_frame)

        self.assertIsInstance(detections_df, pd.DataFrame)
        self.assertIsInstance(detections_df_finetuning, pd.DataFrame)
        self.assertIsInstance(classification_df_finetuning, list)
        self.assertIsInstance(detections_df_subtraction, pd.DataFrame)
        self.assertIsInstance(detections_df_edgedetection, pd.DataFrame)


@patch('cv2.VideoCapture')
@patch('cv2.namedWindow')
@patch('cv2.resizeWindow')
@patch('cv2.imshow')
@patch('cv2.waitKey')
@patch('cv2.destroyAllWindows')
@patch('cv2.moveWindow')
@patch('src.detection.objet_detection.detection_yolov11')
@patch('src.detection.objet_detection.detection_yolov11_fine_tuning')
@patch('src.detection.objet_detection.classification_fine_tuning')
@patch('src.detection.objet_detection.extract_camera_data')
@patch('src.detection.objet_detection.draw_detections')
@patch('src.detection.objet_detection.draw_classification')
@patch('cv2.cvtColor')
@patch('src.detection.objet_detection.background_subtraction')
@patch('src.detection.objet_detection.background_subtraction_on_edges')
@patch('src.detection.objet_detection.detection_windows')
class TestVideoProcessing(unittest.TestCase):
    def test_process_video(self, mock_detection_windows, mock_background_subtraction_on_edges, mock_background_subtraction,
                           mock_cvtColor, mock_draw_classification, mock_draw_detections, mock_extract_camera_data,
                           mock_classification, mock_fine_tuning, mock_detection, mock_move_window, mock_destroy,
                           mock_wait, mock_imshow, mock_resize, mock_named, mock_capture):
        """
        Test the process_video function.

        This test verifies that the process_video function correctly processes a video
        and calls the expected detection and classification functions.
        """
        # Mock de la capture vidéo
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Frame vide avec 3 canaux (RGB)
        mock_cap.read.side_effect = [(True, mock_frame), (True, mock_frame), (False, None)]
        mock_capture.return_value = mock_cap
        mock_extract_camera_data.return_value = (4, "2025-03-22 12:00:00")

        # Mock des détections et classifications
        mock_detection.return_value = pd.DataFrame(
            {'xmin': [0], 'ymin': [0], 'xmax': [100], 'ymax': [100], 'name': ['object'], 'confidence': [0.9]})
        mock_fine_tuning.return_value = pd.DataFrame(
            {'xmin': [10], 'ymin': [10], 'xmax': [90], 'ymax': [90], 'name': ['object'], 'confidence': [0.95]})
        mock_classification.return_value = [MagicMock(class_name='empty', confidence=0.85)]

        # Mock de la détection des fenêtres
        mock_detection_windows.return_value = []  # Pas de fenêtres détectées dans ce test simplifié

        # Mock de la conversion de la couleur
        mock_cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

        # Appel à la fonction à tester
        process_video('fake_video.mp4', nb_of_img_skip_between_2=0)

        # Vérifications : on vérifie que certaines fonctions ont été appelées
        mock_capture.assert_called_once_with('fake_video.mp4')
        self.assertEqual(mock_cap.read.call_count, 3)
        mock_detection.assert_called()
        mock_fine_tuning.assert_called()
        mock_classification.assert_called()
        mock_background_subtraction.assert_called()
        mock_background_subtraction_on_edges.assert_called()
        mock_detection_windows.assert_called()
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
        mock_listdir.return_value = ['video1.mp4', 'video2.mp4']  # Simulate a folder with 2 video files(c'est juste un similation, donc t'est pas obliger d'avoir ces videos, moulaye)

        process_videos('fake_folder', nb_of_img_skip_between_2=0)

        mock_listdir.assert_called_once_with('fake_folder')
        self.assertEqual(mock_process_video.call_count, 2)
        mock_process_video.assert_any_call(os.path.join('fake_folder', 'video1.mp4'), 0)
        mock_process_video.assert_any_call(os.path.join('fake_folder', 'video2.mp4'), 0)

if __name__ == '__main__':
    unittest.main()
