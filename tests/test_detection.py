import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# les fonctions aa tester
from src.detection.objet_detection import process_frame, process_video, process_videos
from src.detection.ai.detection import detection_yolov11
from src.detection.ai.detection_finetuning import detection_yolov11_fine_tuning
from src.detection.ai.classification_finetuning import classification_fine_tuning
from src.detection.light.lowlight_test import enhance_image
from src.detection.utils.utils import draw_rectangle, draw_text

class TestDetection(unittest.TestCase):

    @patch('src.detection.ai.detection.detection_yolov11')
    @patch('src.detection.ai.detection_finetuning.detection_yolov11_fine_tuning')
    @patch('src.detection.ai.classification_finetuning.classification_fine_tuning')
    def test_process_frame(self, mock_classification, mock_fine_tuning, mock_detection):
        
        mock_frame = MagicMock()

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

class TestVideoProcessing(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('src.detection.ai.detection.detection_yolov11')
    @patch('src.detection.ai.detection_finetuning.detection_yolov11_fine_tuning')
    @patch('src.detection.ai.classification_finetuning.classification_fine_tuning')
    def test_process_video(self, mock_classification, mock_fine_tuning, mock_detection, mock_destroy, mock_wait, mock_imshow, mock_resize, mock_named, mock_capture):

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, MagicMock()), (True, MagicMock()), (False, None)]  
        mock_capture.return_value = mock_cap

        process_video('fake_video.mp4', nb_of_img_skip_between_2=0)

        # verifierr
        mock_capture.assert_called_once_with('fake_video.mp4')  # (Verifie que la vidio est ouverte)
        self.assertEqual(mock_cap.read.call_count, 3) 
        mock_detection.assert_called() 
        mock_fine_tuning.assert_called() 
        mock_classification.assert_called() 
        mock_destroy.assert_called_once()  

class TestVideosProcessing(unittest.TestCase):

    @patch('os.listdir')
    @patch('test_detection.process_video')
    def test_process_videos(self, mock_process_video, mock_listdir):
    
        mock_listdir.return_value = ['video1.mp4', 'video2.mp4']  # (Simule un dossier avec 2 fichiers vidéo)

        process_videos('fake_folder', nb_of_img_skip_between_2=0)

        #tester (verifier)
        mock_listdir.assert_called_once_with('fake_folder') 
        self.assertEqual(mock_process_video.call_count, 2) 
        mock_process_video.assert_any_call(os.path.join('fake_folder', 'video1.mp4'), 0)
        mock_process_video.assert_any_call(os.path.join('fake_folder', 'video2.mp4'), 0)

if __name__ == '__main__':
    unittest.main()