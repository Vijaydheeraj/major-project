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

    window_name = f"Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 360)

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
        detections_df, detection_em = process_frame(frame)

        # Show results in video
        for index, row in detections_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            color = (0, 255, 0)  # Green for detected objects
            draw_rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_text(frame, f"{row['name']} ({row['confidence']:.2f})", (x1, y1 - 10), color, 0.5, 2)

        # Show result process by empty_or_not
        if detection_em[0].class_name == 'empty':
            draw_text(frame, f"{detection_em[0].class_name} ({detection_em[0].confidence:.2f})", (10, 30),
                      (0, 255, 0), 0.5,2)
        elif detection_em[0].class_name == 'full':
            draw_text(frame, f"{detection_em[0].class_name} ({detection_em[0].confidence:.2f})", (10, 30),
                      (0, 0, 255), 0.5,2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame: Any) -> tuple[DataFrame, list]:
    """
    Process a single frame for object detection.

    Args:
        frame (Any): The frame to process.

    Returns:
        pd.DataFrame: A DataFrame containing the detection results.
        list: A list containing the empty detection results.
    """
    detections_df = detection_yolov11(frame)
    detections_df_fine_tuning = detection_yolov11_fine_tuning(frame)
    classification_df_finetuning = classification_fine_tuning(frame)

    return detections_df_fine_tuning, classification_df_finetuning

