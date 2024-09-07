from openai import OpenAI
import json
import os
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from PIL import Image
import io
import boto3
import requests

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
s3 = boto3.client('s3', aws_access_key_id=st.secrets['aws_access_key_id'], aws_secret_access_key=st.secrets['aws_secret_access_key'])

bucket = st.secrets["bucket_image_dwls"]
prefix = "web-images/"

def upload_image_to_s3(image, file_name, bucket, prefix):
    final_name = file_name
    s3_key = f"{prefix}{final_name}"
    try:
        # Convertir la imagen a bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Subir la imagen a S3
        s3.upload_fileobj(img_byte_arr, bucket, s3_key)
        st.success("Imagen subida correctamente!")
    except Exception as e:
        st.error(f"Error al subir la imagen: {e}")


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages, # type: ignore
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content 

#prompt = "Escribeme un codigo para dada una lista en python con n+1 numeros, uno de ellos repetidos, poder identificar el numero repetido"
prompt = "Escribeme una frase de matematica o estadistica"
##answer = get_completion(prompt)

text = answer
# Título de la aplicación
st.title("Aplicación de dos columnas con imagen y texto")
# Crear dos columnas
col1, col2 = st.columns([7, 3])

# Cargar imagen en la primera columna
with col1:
    st.header("Imagen")
    uploaded_image = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        file_name = uploaded_image.name
        st.image(image, caption='Imagen subida', use_column_width=True)
        enviar = st.button("Subir imagen a S3")
        #upload_image_to_s3(image, file_name, bucket, prefix)
        if enviar:
            image_url = f"https://{bucket}.s3.eu-north-1.amazonaws.com/{prefix}{file_name}"
            data = {'imageUrl': image_url}
            api_url = st.secrets["api_url"]
            #response = requests.post(api_url, json=data)
            if response.status_code == 200:
                result = response.json()
                st.write(result)
            else:
                st.write("Something went wrong")
        
    else:
        st.text("Por favor, sube una imagen.")

# Añadir textos en la segunda columna
with col2:
    st.header("Texto")
    st.write("Aquí puedes poner algunos textos.")
    st.write("Este es un ejemplo de cómo puedes organizar el contenido en columnas.")
    st.write("¡Puedes añadir más detalles según lo necesites!")
    st.write(text)
