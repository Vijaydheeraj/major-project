# Projet Collectif

## Description

Ce projet à pour but de détecter la présence d'objets dans le tramway afin de savoir s'il est, ou non, vide. 
Ce projet inclut également des configurations chargées à partir d'un fichier JSON 
et des tests unitaires pour vérifier le bon fonctionnement des routes et des configurations.

## Structure du projet

- `app/` : contient les fichiers de l'application Flask (cette partie a été mise en pause et n'est pas utilisée 
dans le projet, elle aurait uniquement été utilisée pour regrouper les fonctions et résultats)
- `config/` : contient les fichiers de configuration
- `detection/` : **contient les différentes fonctions de détection d'objets**
  - `ai/` : **contient les fichiers associées à l'IA de détection d'objets**
    - `classification_finetuning.py` : fichier utilisant l'IA pour la classification : vide ou plein
    - `detection.py` : fichier utilisant YOLO pour la détection d'objets
    - `detection_finetuning.py` : fichier utilisant YOLO fine-tuné pour la détection d'objets
  - `background_substraction/` : **contient les fichiers associées à la soustraction de fond**
    - `background_sub.py` : fichier utilisant la soustraction de pixels et la détection de contours
  - `light/` : **contient les fichiers d'amélioration de luminosité**
    - `ai/` : contient les fichiers associés à l'IA pour l'amélioration de luminosité (code trouvé sur internet).
De nombreux fichiers sont disponibles (apprentissage et utilisation), pour l'utiliser, il suffit d'avoir le modèle dans 
le dossier snapshots et la fonction `enhance_image` dans le fichier `lowlight_test.py`
    - `equalization/` : contient les fichiers de l'égalisation de l'histogramme
      - `light_fast.py` : fichier utilisant l'égalisation de l'histogramme
  - `utils/` : contient un fichier de fonctions utilitaires (affichage, etc.)
    - `utils.py` : fichier contenant des fonctions utilitaires
  - `windows/` : **contient les fichiers associées à la détection de fenêtres**
    - `ai/` : contient les fichiers associés à l'IA pour la détection de fenêtres
      - `windows_finetuning.py` : fichier utilisant l'IA fine-tuné pour la détection de fenêtres
    - `manual/` : contient les fichiers associées à la détection manuelle de fenêtres
      - `windows.py` : fichier contenant la position brute des fenêtres et l'utilisant pour la détection
  - `object_detection.py` : **regroupe l'utilisation des différentes fonctions de détection**
- `unit_tests/` : contient les tests unitaires
- `main.py` : **fichier principal du projet à exécuter**

Si une partie vous intéresse plus particulièrement, vous pouvez :
- Consulter le fichier `objet_detection.py` pour voir la manière dont nous l'utilisons.
- Consulter les fichiers dans le dossier correspondant.

## Prérequis

- Python 3.12
- bibliothèques Python : cf. `requirements.txt`

## Configuration

Afin de faire fonctionner ce projet, un fichier config.json doit être présent dans le dossier `config/`.
Nous ne fournissons pas les variables `roboflow_api_key` et `model_id` puisque nous utilisons l'API de Roboflow pour la 
détection d'objets qui nous limite dans le nombre de requêtes par mois.
Concernant la variable `path`, il s'agit du chemin vers le dossier contenant les vidéos à analyser (les vidéos doivent 
être en couleur).
Ce fichier doit contenir les informations suivantes :

```json
{
  "videos": {
        "path": "path/to/videos",
        "extensions": ["mp4"]
    },

    "ai-detection": {
        "roboflow_api_key": "API OF THE MODEL",
        "model_id": "ID OF THE MODEL"
    },

    "ai-empty": {
        "roboflow_api_key": "API OF THE MODEL",
        "model_id": "ID OF THE MODEL"
    },

    "ai-windows": {
        "roboflow_api_key": "API OF THE MODEL",
        "model_id": "ID OF THE MODEL"
    }
}
```

## Exécution

Il est nécessaire d'avoir suivi les étapes précédentes (Prérequis et Configuration) pour exécuter le projet.
Pour lancer le projet, il suffit d'exécuter le fichier `main.py`.