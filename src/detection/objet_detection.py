import os
import sys
import json
# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
import cv2
import pandas as pd
import numpy as np
from typing import Any
from src.detection.model import model
from src.detection.light import model as light_model
from PIL import Image
import torch

# Initialize the low-light enhancement model
DCE_net = light_model.enhance_net_nopool().cuda()
DCE_net.load_state_dict(torch.load('C:/Users/Siddikh/Documents/Projet_collectif/object_detection/src/detection/light/snapshots/Epoch99.pth'))
#utiliser la variable a la place au lieu du chemin entier
def enhance_image(image):
    data_lowlight = (np.asarray(image) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight)
    
    enhanced_image = enhanced_image.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)

def process_videos(folder_path: str) -> None:
    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        if not filename.endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, filename)
        process_video(video_path)

def process_video(video_path: str) -> None:
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Erreur: Impossible d'ouvrir la vidéo {video_path}.")

     # Obtenir les propriétés de la vidéo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Créer un writer pour sauvegarder la vidéo améliorée
    output_path = video_path.replace('.mp4', '_enhanced.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Créer des fenêtres pour afficher les vidéos
    window_name_original = "Video Originale"
    window_name_enhanced = "Video Améliorée"
    cv2.namedWindow(window_name_original, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_enhanced, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_original, 640, 360)
    cv2.resizeWindow(window_name_enhanced, 640, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fin de la vidéo {video_path} ou erreur de lecture.")
            break

        # Convertir le frame en image PIL
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Améliorer l'image
        enhanced_frame = enhance_image(frame_image)

        # Convertir l'image améliorée en frame OpenCV
        enhanced_frame = cv2.cvtColor(np.array(enhanced_frame), cv2.COLOR_RGB2BGR)

        # Traiter l'image et obtenir les résultats
        detections = process_frame(enhanced_frame)

        # Afficher les résultats dans la vidéo
        for index, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            color = (0, 255, 0)  # Vert pour les objets détectés
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Sauvegarder le frame amélioré dans le fichier vidéo
        out.write(enhanced_frame)

        # Afficher les frames dans les fenêtres
        cv2.imshow(window_name_original, frame)
        cv2.imshow(window_name_enhanced, enhanced_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_frame(frame: Any) -> pd.DataFrame:
    # # Ameliore l'image pour une meilleure detection
    # frame = enhance_image(frame)

    # Détecter les objets dans l'image
    results = model(frame)

    # Récupérer les boîtes englobantes et les confiances
    detections = results[0].boxes.data.cpu().numpy()

    # Convertir les résultats en DataFrame
    detections_df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
    
    # Ajouter les noms des classes
    detections_df['name'] = detections_df['class'].apply(lambda x: model.names[int(x)])

    # Filtrer les objets indésirables
    # TODO: Contraignant --> trouver une solution de remplacement
    detections_df = detections_df[~detections_df['name'].isin(["couch", "chair", "car", "bench", "train"])]

    return detections_df

process_videos('C:/Users/Siddikh/Videos/ProjetCollectif/prise_2')