
# Projet de Reconnaissance Faciale

## 1. Aperçu

Ce projet de reconnaissance faciale utilise le deep learning pour détecter et reconnaître les visages dans des images ou des flux vidéo en temps réel. L'objectif est de développer une application capable de détecter et de reconnaître les visages, puis de les associer à un nom à partir d'une base de données d'images. Cette application utilise les modèles de détection de visages d'OpenCV et de modèles d'encodage de visages pré-entraînés pour générer des vecteurs d'embeddings, qui sont ensuite utilisés pour la reconnaissance.

Le projet se divise en plusieurs étapes :

1. **Extraction des embeddings de visages** : Collecte des données d'embeddings à partir des images.
2. **Entraînement du modèle de reconnaissance** : Formation d'un classificateur pour prédire l'identité des visages.
3. **Reconnaissance en temps réel** : Identification des personnes dans une vidéo en direct, avec la mise à jour de leur présence dans une feuille de calcul Excel.

## 2. Prérequis

Le projet nécessite plusieurs bibliothèques Python et fichiers modèles pour fonctionner :

- **imutils** : pour le traitement d'images.
- **opencv-python** : pour la détection et l'encodage des visages avec des réseaux neuronaux convolutifs.
- **scikit-learn** : pour l'entraînement du modèle.
- **openpyxl** : pour manipuler des fichiers Excel pour la gestion de l'assiduité.
- **pickle** : pour la sérialisation des modèles et des données.

## 3. Structure du Projet

### 3.1 Extraction des Empreintes Faciales

Le script `extract_embeddings.py` extrait des vecteurs d'empreintes faciales uniques (128 dimensions) à partir d'un ensemble d'images. Ces vecteurs servent ensuite à entraîner le modèle de classification pour la reconnaissance faciale.

**Détails Techniques :**

- Le modèle de détection des visages est basé sur un réseau neuronal convolutif (CNN) pré-entraîné utilisant le modèle Caffe `res10_300x300_ssd_iter_140000.caffemodel`.
- L'extraction des empreintes utilise le modèle OpenFace (`openface_nn4.small2.v1.t7`), qui produit des représentations compactes et discriminatives des visages.
- Des techniques de prétraitement comme la normalisation des images et l’ajustement des dimensions sont utilisées pour garantir des performances constantes.

### 3.2 Entraînement du Modèle de Reconnaissance

Le script `train_model.py` entraîne un classificateur SVM (Support Vector Machine) en utilisant les empreintes faciales extraites.

**Détails Techniques :**

- Les vecteurs d'empreintes sont transformés en étiquettes numériques à l'aide d'un encodeur de labels.
- Le classificateur SVM est entraîné avec un noyau linéaire pour séparer efficacement les classes (individus).
- Le modèle entraîné est ensuite sauvegardé pour une utilisation ultérieure.
- Une validation croisée est réalisée pour évaluer la performance du modèle et minimiser les erreurs de classification.

### 3.3 Reconnaissance en Temps Réel et Gestion des Présences

Le script `recognize_video.py` détecte les visages en temps réel à partir d'un flux vidéo et les associe au modèle entraîné pour identifier les individus. Les présences sont ensuite enregistrées dans un fichier Excel.

**Détails Techniques :**

- Le système utilise une webcam pour capturer les flux vidéo.
- Les visages détectés sont comparés aux empreintes enregistrées pour déterminer l'identité.
- Le fichier Excel est automatiquement mis à jour pour indiquer la présence ou l'absence des individus.
- Le script inclut une gestion des seuils de confiance pour minimiser les faux positifs et négatifs.

## 4. Fonctionnalités Clés

1. **Extraction des Empreintes Faciales :**
   - Utilise un détecteur de visages basé sur des réseaux neuronaux convolutifs.
   - Génère des vecteurs de 128 dimensions représentant de manière unique les caractéristiques faciales.
   - Prétraitement des images pour assurer une extraction fiable dans des conditions d’éclairage variées.

2. **Modèle de Reconnaissance :**
   - Entraîné à l'aide d'un SVM sur les empreintes faciales extraites.
   - Fournit des prédictions probabilistes pour classer les visages détectés.
   - Vérification de la précision avec des métriques comme la précision, le rappel et le F1-score.

3. **Reconnaissance en Temps Réel :**
   - Traite des flux vidéo pour détecter et reconnaître les visages en temps réel.
   - Intègre les résultats dans un fichier Excel pour faciliter la gestion des présences.
   - Capacité à gérer des bases de données de visages de grande taille.

## 5. Dépendances

- **Bibliothèques Python :**
  - OpenCV
  - imutils
  - numpy
  - scikit-learn
  - openpyxl

- **Matériel :** Webcam pour la capture vidéo.

- **Modèles Pré-entraînés :**
  - Détecteur de visages d'OpenCV (`res10_300x300_ssd_iter_140000.caffemodel`)
  - Modèle d'empreintes faciales OpenFace (`openface_nn4.small2.v1.t7`)

## 6. Exemple de Résultat

[Insérer ici des captures d'écran ou des exemples de résultats]

## 7. Améliorations Futures

- **Reconnaissance Multi-Visages :** Améliorer la détection pour gérer plusieurs visages par image.
- **Intégration Mobile :** Déployer le système sur des plateformes mobiles pour une accessibilité plus large.
- **Précision Améliorée :** Intégrer des modèles avancés comme FaceNet, DLIB, ou des architectures modernes de Deep Learning (ex : ResNet) pour une meilleure précision.
- **Sécurité et Vie Privée :** Mettre en œuvre des méthodes de cryptage pour protéger les données biométriques.

## 8. Conclusion

Ce projet montre comment utiliser le deep learning pour la détection et la reconnaissance faciale avec OpenCV et des modèles pré-entraînés. L'application peut être étendue à diverses applications pratiques, telles que la gestion de l'assiduité, le contrôle d'accès, et même l'analyse de sécurité. Le modèle peut être amélioré avec des données supplémentaires pour augmenter la précision, et des optimisations peuvent être effectuées pour le rendre plus rapide et plus robuste en conditions réelles.

Le processus complet inclut la collecte des données, l'entraînement du modèle, et l'application en temps réel, le tout utilisant des techniques de deep learning pour garantir une haute précision et des performances robustes.
