from openai import OpenAI
import json
import os
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from PIL import Image


client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages, # type: ignore
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content 

prompt = "Escribeme un codigo para dada una lista en python con n+1 numeros, uno de ellos repetidos, poder identificar el numero repetido"
answer = get_completion(prompt)

text = answer
# Título de la aplicación
st.title("Aplicación de dos columnas con imagen y texto")
# Crear dos columnas
col1, col2 = st.columns(2)

# Cargar imagen en la primera columna
with col1:
    st.header("Imagen")
    uploaded_image = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Imagen subida', use_column_width=True)
    else:
        st.text("Por favor, sube una imagen.")

# Añadir textos en la segunda columna
with col2:
    st.header("Texto")
    st.write("Aquí puedes poner algunos textos.")
    st.write("Este es un ejemplo de cómo puedes organizar el contenido en columnas.")
    st.write("¡Puedes añadir más detalles según lo necesites!")
    st.write(text)
