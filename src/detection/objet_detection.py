import os

# Ajouter le répertoire racine au PYTHONPATH
# TODO : Retirer les 2 lignes si fonctionne sur les autres machines
#import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import pandas as pd
import numpy as np
from typing import Any
from src.detection.model import model
from src.detection.light import model as light_model
from PIL import Image
import torch

# Initialize the low-light enhancement model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'light', 'snapshots', 'epoch99.pth')
DCE_net = light_model.enhance_net_nopool()
DCE_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def enhance_image(frame):
    # TODO : Déplacer la fonction dans un autre fichier
    if not isinstance(frame, np.ndarray):
        raise TypeError("Le frame fourni n'est pas un tableau NumPy. Vérifiez la source de l'image.")
    # Convertir le frame en image PIL
    frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Améliorer l'image
    data_lowlight = (np.asarray(frame_image) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight)
    
    enhanced_image = enhanced_image.squeeze().permute(1, 2, 0).numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    enhanced_frame = Image.fromarray(enhanced_image)

    # Convertir l'image améliorée en frame OpenCV
    enhanced_frame = cv2.cvtColor(np.array(enhanced_frame), cv2.COLOR_RGB2BGR)

    # Traiter l'image et obtenir les résultats
    #detections = process_frame(enhanced_frame) # supprimer cette ligne si tu veux pas calculer sur l'image amélioree

    # Afficher les résultats dans la vidéo #ce paragraphe aussi
    '''for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        color = (0, 255, 0)  # Vert pour les objets détectés
        cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(enhanced_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)'''

    return enhanced_frame

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
    # TODO : Mettre dans une fonction utilitaire dans utils.py et ne plus l'appeler ici
    output_path = video_path.replace('.mp4', '_enhanced.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Créer des fenêtres pour afficher les vidéos
    # TODO : N'afficher que la video amelioree ou la video originale (choix a faire)
    #  avec le traitement de l'image fait dessus en appelant process_frame (après l'appel a la fonction enhance_image)
    window_name_original = "video originale"
    window_name_enhanced = "video amelioree"
    cv2.namedWindow(window_name_original, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_enhanced, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_original, 640, 360)
    cv2.resizeWindow(window_name_enhanced, 640, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fin de la vidéo {video_path} ou erreur de lecture.")
            break

        # Améliorer l'image
        enhanced_frame = enhance_image(frame)

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

