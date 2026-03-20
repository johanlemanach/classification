# Satellite Image Classification - MLOps & Monitoring (E5)

Ce projet est une application web Flask dédiée à la classification d'images satellite (4 classes : desert, forest, meadow, mountain) utilisant un modèle de Deep Learning (Keras/CNN). 

Il a été développé dans le cadre de la certification **E5 (Cas pratique C20, C21)** pour démontrer la mise en place d'un cycle de vie MLOps complet, incluant le monitorage, l'alerting et la résolution d'incidents techniques.

## 🚀 Fonctionnalités principales

*   **Classification en temps réel :** Upload d'images satellite et prédiction via un modèle CNN pré-entraîné.
*   **Monitorage applicatif (C20) :**
    *   Endpoint `/metrics` exposant les compteurs de requêtes, erreurs et latences au format JSON.
    *   Intégration de `flask-monitoringdashboard` pour une visualisation détaillée des performances.
    *   Système d'alerting basé sur des seuils (latence > 800ms, confiance < 0.55).
*   **Boucle de Feedback (MLOps) :**
    *   Possibilité pour l'utilisateur de corriger une prédiction erronée.
    *   Stockage du feedback (image, prédiction, correction) dans une base de données SQLite pour futur réentraînement.
*   **Journalisation structurée :** Séparation des logs applicatifs (`app.log`) et des alertes critiques (`alerts.log`).

## 🛠️ Installation et Utilisation

### Prérequis
*   Python 3.8+
*   Environnement virtuel recommandé

### Installation
```bash
# Cloner le dépôt
git clone git@github.com:johanlemanach/classification.git
cd classification

# Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate sur Windows

# Installer les dépendances
python -m pip install --upgrade pip
python -m pip install -r app/requirements.txt
```

### Lancement
```bash
python app/app.py
```
L'application est accessible sur `http://127.0.0.1:5000`.
Le dashboard de monitorage local est disponible sur `http://127.0.0.1:5000/dashboard`.

## 🧪 Tests et Qualité (C21)

Le projet inclut une suite de tests unitaires et d'intégration pour garantir la robustesse de l'inférence et du monitorage.

```bash
python -m unittest discover -s app/tests -v
```

La sortie attendue est `Ran 4 tests ... OK`.

### Incidents techniques résolus
L'application a fait l'objet de corrections majeures pour assurer la précision des prédictions :
1.  **Correction de la normalisation :** Suppression de la double division par 255 (le modèle possède sa propre couche de rescaling).
2.  **Redimensionnement d'entrée :** Harmonisation systématique des images en 224x224 pour correspondre à l'architecture du CNN.

## 📈 Architecture du Projet
*   `app/` : Source de l'application Flask.
*   `app/models/` : Modèle Keras `.keras` utilisé pour l'inférence.
*   `app/instance/` : Base de données SQLite pour le feedback (ignorée par Git).
*   `app/instance/flask_monitoringdashboard.db` : Base SQLite du dashboard local.
*   `app/logs/` : Journaux d'événements et d'alertes (ignorés par Git).
*   `images_to_test/` : Jeu de données pour tester l'application manuellement.

---
*Projet réalisé pour la certification Développeur en Intelligence Artificielle.*
