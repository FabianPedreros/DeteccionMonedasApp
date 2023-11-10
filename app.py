import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

!sudo apt-get update
!sudo apt-get install -y libgl1-mesa-glx

model = YOLO("best.pt")

st.title('Deteccion de plantas de tomate')
st.header('Aplicacion para la deteccion de plantas de tomate')

st.write('Carga una imagen desde tu dispositivo para detectar si hay una planta de tomate')

# Cargar la imagen desde la interfaz de la cámara
foto = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

if foto is not None:
    # Convertir la imagen a un formato compatible con YOLO
    img = Image.open(foto)
    img = np.array(img)
    
    # Realizar la predicción con YOLO
    results = model.predict(img, imgsz=412, conf=0.5)
    
    # Mostrar la imagen original con las cajas detectadas
    st.image(results[0].plot(labels=True), caption="Resultado de la detección", use_column_width=True)

    # Inicializar la lista de clases halladas
    clases_halladas = []

    for result in results:
        clases_halladas.extend(result.boxes.cls.numpy())

    st.write(clases_halladas)

    mapeo = {
        0: 'Planta de tomate',
        1: 'Planta de albahaca',
        2: 'Planta de lechuga'
    }

    # Utiliza Counter para contar las ocurrencias de cada elemento en el arreglo
    conteo = Counter(clases_halladas)

    # Imprime el resultado
    for clase, cantidad in conteo.items():
        nombre_clase = mapeo.get(clase, f'Clase Desconocida ({clase})')
        st.write(f"La cantidad de {nombre_clase} detectadas es: {cantidad}")
