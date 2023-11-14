import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

import subprocess

# Ejecutar 'sudo apt-get update'
#subprocess.run(['sudo', 'apt-get', 'update'])

# Ejecutar 'sudo apt-get install -y libgl1-mesa-glx'
#subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libgl1-mesa-glx'])

#model = YOLO("best.pt")
model = YOLO("bestMonedas.pt")

st.title('Deteccion de monedas')
st.header('Aplicacion para la deteccion de monedas')

st.write('Carga una imagen desde tu dispositivo para detectar las monedas y calcular el monto total')

# Cargar la imagen desde la interfaz de la cámara
foto = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

if foto is not None:
    # Convertir la imagen a un formato compatible con YOLO
    img = Image.open(foto)
    img = np.array(img)
    
    # Realizar la predicción con YOLO
    results = model.predict(img, imgsz=416, conf=0.5)
    
    # Mostrar la imagen original con las cajas detectadas
    st.image(results[0].plot(labels=True), caption="Resultado de la detección", use_column_width=True)

    # Inicializar la lista de clases halladas
    clases_halladas = []

    for result in results:
        clases_halladas.extend(result.boxes.cls.numpy())

    #st.write(results)

    #st.write(clases_halladas)

    mapeo = {
        0: 100, 
        1: 1000, 
        2: 200, 
        3: 50, 
        4: 500
    }

    # Utiliza Counter para contar las ocurrencias de cada elemento en el arreglo
    conteo = Counter(clases_halladas)

    montoTotal = 0
    # Imprime el resultado
    for clase, cantidad in conteo.items():
        nombre_clase = mapeo.get(clase, f'Clase Desconocida ({clase})')
        montoTotal = montoTotal + (nombre_clase*cantidad)
        st.write(f"La cantidad de monedas de **${nombre_clase}** pesos detectadas es: **{cantidad}**")

    st.write(f"El monto de las monedas detectadas es de **${montoTotal}** pesos colombianos")
