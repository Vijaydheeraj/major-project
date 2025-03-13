import os
import sys
import cv2
import pandas as pd
from typing import Any

from pandas import DataFrame

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.detection.ai.detection import detection_yolov11
from src.detection.ai.detection_finetuning import detection_yolov11_fine_tuning
from src.detection.ai.classification_finetuning import classification_fine_tuning
from src.detection.utils.utils import draw_rectangle, draw_text, extract_camera_data
from src.detection.background_substraction.background_sub import background_substraction
from src.detection.utils.utils import draw_rectangle, draw_text


def process_videos(folder_path: str, nb_of_img_skip_between_2: int=0) -> None:
    """
    Process all video files in the specified folder.

    Args:
        folder_path (str): The path to the folder containing video files.
        nb_of_img_skip_between_2 (int): Number of images to skip between 2 images. Defaults to 0.

    Returns:
        None
    """
    for filename in os.listdir(folder_path):
        if not filename.endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, filename)
        process_video(video_path, nb_of_img_skip_between_2)


def process_video(video_path: str, nb_of_img_skip_between_2: int) -> None:
    """
    Process a single video file for object detection.
    Opens a window and displays the result.

    Args:
        video_path (str): The path to the video file.
        nb_of_img_skip_between_2 (int): Number of images to skip between 2 images.

    Raises:
        IOError: If the video file cannot be opened.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Erreur: Impossible d'ouvrir la vidéo {video_path}.")

    window_name_1 = "Detections"
    window_name_2 = "Fine-tuning and Classification"
    cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_1, 640, 360)
    cv2.resizeWindow(window_name_2, 640, 360)
    cv2.moveWindow(window_name_1, 0, 0)
    cv2.moveWindow(window_name_2, 650, 0)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fin de la vidéo {video_path} ou erreur de lecture.")
            break

        # Skip some images
        frame_count += 1
        if frame_count % (nb_of_img_skip_between_2 + 1) != 0:
            continue

        # Image processing and results
        camera_number, time_str = extract_camera_data(video_path)
        detections_df, detections_df_finetuning, classification_df_finetuning = process_frame(frame, camera_number)

        # Create copies of the frame for each window
        frame_detections = frame.copy()
        frame_finetuning = frame.copy()

        # Show results in the first window
        for index, row in detections_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            color = (0, 255, 0)  # Green for detected objects
            draw_rectangle(frame_detections, (x1, y1), (x2, y2), color, 2)
            draw_text(frame_detections, f"{row['name']} ({row['confidence']:.2f})", (x1, y1 - 10), color, 0.5, 2)

        # Show results in the second window
        for index, row in detections_df_finetuning.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            color = (0, 255, 0)  # Green for detected objects
            draw_rectangle(frame_finetuning, (x1, y1), (x2, y2), color, 2)
            draw_text(frame_finetuning, f"{row['name']} ({row['confidence']:.2f})", (x1, y1 - 10), color, 0.5, 2)

        # Show result process by empty_or_not
        if classification_df_finetuning[0].class_name == 'empty':
            draw_text(frame_finetuning,
                      f"{classification_df_finetuning[0].class_name} ({classification_df_finetuning[0].confidence:.2f})",
                      (10, 30),
                      (0, 255, 0), 0.5, 2)
        elif classification_df_finetuning[0].class_name == 'full':
            draw_text(frame_finetuning,
                      f"{classification_df_finetuning[0].class_name} ({classification_df_finetuning[0].confidence:.2f})",
                      (10, 30),
                      (0, 0, 255), 0.5, 2)

        cv2.imshow(window_name_1, frame_detections)
        cv2.imshow(window_name_2, frame_finetuning)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame: Any, camera_number: int) -> tuple[DataFrame, DataFrame, list]:
    """
    Process a single frame for object detection.

    Args:
        frame (Any): The frame to process.
        camera_number (int): The index of the camera.

    Returns:
        pd.DataFrame: A DataFrame containing the detection results.
        pd.DataFrame: A DataFrame containing the detection results after fine tuning.
        list: A list containing the empty detection results.
    """
    detections_df = detection_yolov11(frame)
    detections_df_fine_tuning = detection_yolov11_fine_tuning(frame)
    classification_df_finetuning = classification_fine_tuning(frame)
    #background_substraction(camera_number, frame)

    return detections_df, detections_df_fine_tuning, classification_df_finetuning

