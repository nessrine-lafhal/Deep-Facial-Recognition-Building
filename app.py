import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import pickle
from imutils.video import VideoStream
import imutils
import time
import csv

# Fonction pour charger les modèles et ressources
def load_resources(detector_path, embedding_model_path, recognizer_path, label_encoder_path):
    protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
    modelPath = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embedder = cv2.dnn.readNetFromTorch(embedding_model_path)
    recognizer = pickle.loads(open(recognizer_path, "rb").read())
    le = pickle.loads(open(label_encoder_path, "rb").read())
    return detector, embedder, recognizer, le

# Fonction pour enregistrer la présence dans un fichier CSV
def save_attendance_to_csv(file_path, detected_names, le):
    fieldnames = ["Name", "Status"]
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name in le.classes_:
                writer.writerow({"Name": name, "Status": "Absent"})

    with open(file_path, mode='r+') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        f.seek(0)
        f.truncate()
        for row in rows:
            if row["Name"] in detected_names:
                row["Status"] = "Present"
            else:
                row["Status"] = "Absent"
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Interface Streamlit avec mise en page professionnelle
st.set_page_config(page_title="Gestion des présences", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        h1 {color: #343a40; text-align: center;}
        .sidebar .sidebar-content {background: #e9ecef;}
        .css-1d391kg {background: #6c757d; color: white;}
        .stButton > button {background-color: #007bff; color: white; border-radius: 5px; border: none;}
    </style>
""", unsafe_allow_html=True)

st.title("Système Administratif de Gestion des Présences")
st.markdown("### Suivi des présences des participants avec reconnaissance faciale")

# Chargement des fichiers
st.sidebar.header("Paramètres")
detector_path = st.sidebar.text_input("Chemin du détecteur", "face_detection_model")
embedding_model_path = st.sidebar.text_input("Chemin du modèle d'embedding", "openface_nn4.small2.v1.t7")
recognizer_path = st.sidebar.text_input("Chemin du modèle de reconnaissance", "output/PyPower_recognizer.pickle")
label_encoder_path = st.sidebar.text_input("Chemin de l'encodeur de labels", "output/PyPower_label.pickle")
csv_file_path = st.sidebar.text_input("Chemin du fichier CSV", "attendance.csv")
confidence_threshold = st.sidebar.slider("Seuil de confiance", 0.0, 1.0, 0.5)

# Téléchargement et affichage du fichier Excel
excel_file_path = st.sidebar.file_uploader("Téléchargez le fichier Excel des participants", type=["xlsx", "xls"])
if excel_file_path:
    st.subheader("Liste des participants")
    df = pd.read_excel(excel_file_path)
    st.dataframe(df)

if "capture_active" not in st.session_state:
    st.session_state.capture_active = False

# Boutons pour démarrer et arrêter le flux vidéo
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Démarrer le flux vidéo"):
        st.session_state.capture_active = True
with col2:
    if st.button("Arrêter le flux vidéo"):
        st.session_state.capture_active = False

if st.session_state.capture_active:
    st.info("Chargement des modèles...")
    detector, embedder, recognizer, le = load_resources(detector_path, embedding_model_path, recognizer_path, label_encoder_path)
    st.success("Modèles chargés avec succès.")

    st.info("Démarrage du flux vidéo...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    detected_names = set()

    FRAME_WINDOW = st.image([])
    while st.session_state.capture_active:
        frame = vs.read()
        frame = imutils.resize(frame, width=640)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                name = le.classes_[j]
                detected_names.add(name)

                text = "{}".format(name)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    vs.stop()
    save_attendance_to_csv(csv_file_path, detected_names, le)
    st.success("Présence enregistrée dans le fichier CSV.")
