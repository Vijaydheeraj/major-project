import os
import sys
import cv2
import pandas as pd
from typing import Any, Optional

from pandas import DataFrame

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.detection.ai.detection import detection_yolov11
from src.detection.ai.detection_finetuning import detection_yolov11_fine_tuning
from src.detection.ai.classification_finetuning import classification_fine_tuning
from src.detection.background_substraction.background_sub import background_subtraction, background_subtraction_on_edges
from src.detection.windows.ai.windows_finetuning import detection_windows, filter_occluded_objects
from src.detection.utils.utils import extract_camera_data, draw_detections, draw_classification

def process_videos(folder_path: str, nb_of_img_to_check: int=1) -> None:
    """
    Process all video files in the specified folder.

    Args:
        folder_path (str): The path to the folder containing video files.
        nb_of_img_to_check (int): Number of images to check. Default : 0.

    Returns:
        None
    """
    frames = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, filename)
        frame = process_video(video_path, nb_of_img_to_check)
        if frame is not None:
            frames.append(frame)

    if frames:
        for i, frame in enumerate(frames):
            cv2.imshow(f"Frame {i}", cv2.resize(frame, (640, 320)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(video_path: str, nb_of_img_to_check: int) -> Optional[Any]:
    """
    Process a single video file for object detection.
    Opens a window and displays the result.

    Args:
        video_path (str): The path to the video file.
        nb_of_img_to_check (int): Number of images to check.

    Raises:
        IOError: If the video file cannot be opened.

    Returns:
        Optional[any]: The last valid frame if the condition is met, otherwise None.
    """
    print(f"Début de la vidéo {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Erreur: Impossible d'ouvrir la vidéo {video_path}.")

    window_name_1 = "Detections"
    window_name_2 = "Fine-tuning and Classification"
    window_name_3 = "Background subtraction"
    window_name_4 = "Edge detection"
    cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_3, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_4, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_1, 640, 360)
    cv2.resizeWindow(window_name_2, 640, 360)
    cv2.resizeWindow(window_name_3, 640, 360)
    cv2.resizeWindow(window_name_4, 640, 360)
    cv2.moveWindow(window_name_1, 0, 0)
    cv2.moveWindow(window_name_2, 650, 0)
    cv2.moveWindow(window_name_3, 0, 390)
    cv2.moveWindow(window_name_4, 650, 390)

    frames_with_objects = []
    threshold = (nb_of_img_to_check // 2) + 1

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fin de la vidéo ou erreur de lecture.")
            break

        # Skip some images
        frame_count += 1
        if frame_count > nb_of_img_to_check:
            break

        # Image processing and results
        camera_number, time_str = extract_camera_data(video_path)
        (detections_df,
         detections_df_finetuning,
         classification_df_finetuning,
         detections_df_subtraction,
         detections_df_edgedetection) = process_frame(frame, camera_number)

        # Create copies of the frame for each window
        frame_detections = frame.copy()
        frame_finetuning = frame.copy()
        frame_subtraction = frame.copy()
        frame_edgedetection = frame.copy()

        # Show results in the first window
        draw_detections(frame_detections, detections_df)

        # Show results in the second window
        draw_detections(frame_finetuning, detections_df_finetuning)
        draw_classification(frame_finetuning, classification_df_finetuning)

        # Show results in the third window
        draw_detections(frame_subtraction, detections_df_subtraction)

        # Show results in the fourth window
        draw_detections(frame_edgedetection, detections_df_edgedetection)

        # Check if there are any objects detected
        if not detections_df_finetuning.empty:
            frames_with_objects.append(frame_finetuning)

        # Affichage
        cv2.imshow(window_name_1, frame_detections)
        cv2.imshow(window_name_2, frame_finetuning)
        cv2.imshow(window_name_3, frame_subtraction)
        cv2.imshow(window_name_4, frame_edgedetection)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Arrêt forcé de la vidéo.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if len(frames_with_objects) >= threshold:
        return frames_with_objects[-1]
    return None


def process_frame(frame: Any, camera_number: int) -> tuple[DataFrame, DataFrame, list, DataFrame, DataFrame]:
    """
    Process a single frame for object detection.

    Args:
        frame (Any): The frame to process.
        camera_number (int): The index of the camera.

    Returns:
        pd.DataFrame: A DataFrame containing the detection results after use of ai (yolo).
        pd.DataFrame: A DataFrame containing the detection results after use of ai with fine tuning.
        list: A list containing the empty detection results.
        pd.DataFrame: A DataFrame containing the detection results after background subtraction.
        pd.DataFrame: A DataFrame containing the detection results after edge detection.
    """

    # Perform window detection
    windows = detection_windows(frame)

    # Perform object detection using YOLO
    detections_df = detection_yolov11(frame)
    detections_df = filter_occluded_objects(detections_df, windows)
    detections_df = detections_df[~detections_df['name'].isin(['couch', 'surfboard', 'train', 'bench', 'chair'])] # Filter unwanted objects (bad solution)

    # Perform object detection using YOLOv1.1 with fine-tuning
    detections_df_fine_tuning = detection_yolov11_fine_tuning(frame)
    detections_df_fine_tuning = filter_occluded_objects(detections_df_fine_tuning, windows)

    # Perform classification using YOLOv1.1 with fine-tuning
    classification_df_finetuning = classification_fine_tuning(frame)

    # Perform background subtraction
    detections_df_subtraction = background_subtraction(camera_number, frame)

    # Perform background subtraction using edge detection
    detections_df_edgedetection = background_subtraction_on_edges(camera_number, frame)

    # Dessiner sur la frame le résultat de la détection des fenêtres
    for polygon in windows:
        points = [(int(point[0]), int(point[1])) for point in polygon.exterior.coords]
        for i in range(len(points)):
            cv2.line(frame, points[i], points[(i + 1) % len(points)], (255, 0, 0), 2)

    return (detections_df, detections_df_fine_tuning,
            classification_df_finetuning, detections_df_subtraction,
            detections_df_edgedetection)
