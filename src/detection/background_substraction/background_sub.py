import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.config.config_loader import load_config, get_video_path
from src.detection.light.lowlight_test import enhance_image


"""
Description: Draws a parallelogram on the given image.
Inputs:
    - image: The image on which to draw the parallelogram (numpy array).
    - pts: The points defining the parallelogram (list of tuples).
    - color: The color of the parallelogram (tuple of BGR values).
    - thickness: The thickness of the parallelogram lines (int).
Outputs: None
"""
def draw_parallelogram(image, pts, color, thickness):
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


"""
Description: Defines the coordinates of the occlusion zones for a given camera.
Inputs:
    - camera: The camera number (4,5,7 or 8).
Outputs:
    - coord: A list of lists, where each inner list contains four tuples representing the coordinates of a parallelogram.
"""
def define_coordinates_parallelograms(camera):
    coord = []
    match camera:
        case 4:
            x1, y1, x2, y2 = 730, 90, 1020, 100
            x3, y3, x4, y4 = 1098, 120, 1250, 138
            x5, y5, x6, y6 = 0, 0, 210, 0
            x7, y7, x8, y8 = 400, 90, 690, 80
            x9, y9, x10, y10 = 565, 150, 700, 150
            x11, y11, x12, y12 = 380, 160, 460, 160
            coord.append([(x1, y1), (x2, y2), (x2, y2 + 330), (x1, y1 + 245)])
            coord.append([(x3, y3), (x4, y4), (x4, y4 + 390), (x3, y3 + 370)])
            coord.append([(x5, y5), (x6, y6), (x6, y6 + 800), (x5, y5 + 800)])
            coord.append([(x7, y7), (x8, y8), (x8, y8 + 70), (x7, y7 + 67)])
            coord.append([(x9, y9), (x10, y10), (x10, y10 + 175), (x9, y9 + 135)])
            coord.append([(x11, y11), (x12, y12), (x12, y12 + 65), (x11, y11 + 60)])
        case 5:
            x1, y1, x2, y2 = 795, 90, 1065, 100
            x3, y3, x4, y4 = 1125, 120, 1275, 155
            x5, y5, x6, y6 = 455, 75, 750, 75
            x7, y7, x8, y8 = 0, 0, 210, 0
            x9, y9, x10, y10 = 620, 130, 655, 130
            x11, y11, x12, y12 = 425, 140, 515, 140
            coord.append([(x1, y1), (x2, y2), (x2, y2 + 325), (x1, y1 + 230)])
            coord.append([(x3, y3), (x4, y4), (x4, y4 + 378), (x3, y3 + 350)])
            coord.append([(x5, y5), (x6, y6), (x6, y6 + 50), (x5, y5 + 60)])
            coord.append([(x7, y7), (x8, y8), (x8, y8 + 800), (x7, y7 + 800)])
            coord.append([(x9, y9), (x10, y10), (x10, y10 + 145), (x9, y9 + 135)])
            coord.append([(x11, y11), (x12, y12), (x12, y12 + 70), (x11, y11 + 35)])
        case 7:
            x1, y1, x2, y2 = 770, 90, 1040, 100
            x3, y3, x4, y4 = 1125, 120, 1265, 147
            x5, y5, x6, y6 = 425, 75, 730, 75
            x7, y7, x8, y8 = 0, 0, 210, 0
            x9, y9, x10, y10 = 610, 130, 645, 130
            x11, y11, x12, y12 = 415, 140, 500, 140
            coord.append([(x1, y1), (x2, y2), (x2, y2 + 320), (x1, y1 + 230)])
            coord.append([(x3, y3), (x4, y4), (x4, y4 + 378), (x3, y3 + 350)])
            coord.append([(x5, y5), (x6, y6), (x6, y6 + 50), (x5, y5 + 60)])
            coord.append([(x7, y7), (x8, y8), (x8, y8 + 800), (x7, y7 + 800)])
            coord.append([(x9, y9), (x10, y10), (x10, y10 + 145), (x9, y9 + 135)])
            coord.append([(x11, y11), (x12, y12), (x12, y12 + 70), (x11, y11 + 35)])
        case 8:
            x1, y1, x2, y2 = 800, 90, 900, 100
            x3, y3, x4, y4 = 675, 95, 744, 100
            x5, y5, x6, y6 = 1085, 140, 1279, 160
            x7, y7, x8, y8 = 901, 100, 1070, 115 
            x9, y9, x10, y10 = 0, 0, 210, 0
            x11, y11, x12, y12 = 355, 90, 524, 80
            coord.append([(x1, y1), (x2, y2), (x2, y2 + 190), (x1, y1 + 180)])
            coord.append([(x3, y3), (x4, y4), (x4, y4 + 174), (x3, y3 + 184)])
            coord.append([(x5, y5), (x6, y6), (x6, y6 + 360), (x5, y5 + 335)])
            coord.append([(x7, y7), (x8, y8), (x8, y8 + 330), (x7, y7 + 295)])
            coord.append([(x9, y9), (x10, y10), (x10, y10 + 800), (x9, y9 + 800)])
            coord.append([(x11, y11), (x12, y12), (x12, y12 + 40), (x11, y11 + 40)])
        case _:
            print("Erreur : Camera non reconnue")
            exit()
    return coord



"""
Description: Performs background subtraction to detect objects in a video frame.
Inputs:
    - camera: The camera number (int).
    - video_test_dir: The directory of the test video (str).
    - video_test_name: The name of the test video (str).
    - frame_tested: The frame number to be tested (int).
Outputs: None
"""
def background_substraction(camera, video_test_dir, video_test_name, frame_tested):
    # Load configuration
    config = load_config()
    video_path = get_video_path(config)

    # Reference video directory name
    video_ref_dir = "prise_0"

    match camera:
        case 4 :
            video_ref_name = "vlc-record-2025-01-23-14h53m38s-rtsp___10.129.52.34_live_ID-STREAM-CAM4H-VID1-.mp4"
        case 5 :
            video_ref_name = "vlc-record-2025-01-23-14h53m36s-rtsp___10.129.52.34_live_ID-STREAM-CAM5H-VID1-.mp4"
        case 7 :
            video_ref_name = "vlc-record-2025-01-23-14h53m24s-rtsp___10.129.52.34_live_ID-STREAM-CAM7H-VID1-.mp4"
        case 8 :
            video_ref_name = "vlc-record-2025-01-23-14h53m33s-rtsp___10.129.52.34_live_ID-STREAM-CAM8H-VID1-.mp4"
        case _ :
            print("Erreur : Camera non reconnue")
            exit()

    # Load the reference video (without objects)
    cap_ref = cv2.VideoCapture(os.path.join(video_path, video_ref_dir, video_ref_name))

    # Load the current video (with added objects)
    cap_current = cv2.VideoCapture(os.path.join(video_path, video_test_dir, video_test_name))

    # Get the 100th frame of the reference video
    cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret_ref, frame_ref = cap_ref.read()
    if not ret_ref:
        print("Erreur : Impossible de lire la vidéo de référence à la frame 100")
        cap_ref.release()
        cap_current.release()
        exit()

    # Luminosity treatment
    frame_ref_light = enhance_image(frame_ref)
    

    # Get the nth frame of the current video
    cap_current.set(cv2.CAP_PROP_POS_FRAMES, frame_tested)
    ret_cur, frame_cur = cap_current.read()
    if not ret_cur:
        print("Erreur : Impossible de lire la vidéo actuelle à la frame " + str(frame_tested))
        cap_ref.release()
        cap_current.release()
        exit()

    # Luminosity treatment
    frame_cur_light = enhance_image(frame_cur)


    # --- ADD A BLUE RECTANGLE FOR EXCLUSION ZONES ---

    coord = define_coordinates_parallelograms(camera)

    # Define the points of the parallelograms
    parallelograms = []
    parallelos = []

    for points in coord:
        parallelogram = np.array(points, np.int32)
        parallelo = points
        parallelograms.append(parallelogram)
        parallelos.append(parallelo)
       

    # --- EXCLUDE WINDOW ZONES ---

    # Convert to grayscale for subtraction
    gray_ref = cv2.cvtColor(frame_ref_light, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(frame_cur_light, cv2.COLOR_BGR2GRAY)

    # Apply image subtraction
    diff = cv2.absdiff(gray_ref, gray_cur)

    # Threshold to detect significant differences (added objects)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Remove window zones (set pixels in this region to zero)
    for parallelogram in parallelograms:
        cv2.fillPoly(thresh, [parallelogram], 0)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect contours of present objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small objects
    min_size = 25  # (minimum size in pixels)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_size ** 2]

    # Draw detected objects on the current image
    output = frame_cur.copy()
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # Draw a blue rectangle on the image to visualize the exclusion zone
    for parallelo in parallelos:
        draw_parallelogram(output, parallelo, (255, 0, 0), 2)


    # Display the result
    cv2.imshow("Objets detectes", output)
    cv2.waitKey(0)

    # Release resources
    cap_ref.release()
    cap_current.release()
    cv2.destroyAllWindows()


# Test

background_substraction(7,"prise_13", "vlc-record-2025-01-23-15h34m49s-rtsp___10.129.52.34_live_ID-STREAM-CAM7H-VID1-.mp4", 1000)
