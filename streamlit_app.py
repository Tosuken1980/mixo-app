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
secret_token = st.secrets['token']
bucket = st.secrets["bucket_image_dwls"]
prefix = "web-images/"

answer = ""
show_result = False
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
        st.success("Upload to S3 successfully!")
    except Exception as e:
        st.error(f"Error uploading: {e}")


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages, # type: ignore
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content 

st.set_page_config(layout="wide")

# Añadir CSS personalizado para cambiar el color de fondo
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #ffb3ba;  /* Cambia esto al color que desees */
    }
    .main .block-container {
        background-color: #a8e6cf;  /* Cambia esto al color que desees */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título de la aplicación
st.title("Cocktail menu evaluation")
# Crear dos columnas
col1, col2 = st.columns([7, 3])

# Cargar imagen en la primera columna
with col1:
    st.header("Imagen")
    uploaded_image = st.file_uploader("Upload a cocktail image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        file_name = uploaded_image.name
        st.image(image, caption='upload image', use_column_width=True)
        upload_image_to_s3(image, file_name, bucket, prefix)
        token = st.text_input("Enter your token:", "")
        if token == secret_token:
            enviar = st.button("Analyze image")
            if enviar:
                with st.spinner("Processing image..."):
                    image_url = f"https://{bucket}.s3.eu-north-1.amazonaws.com/{prefix}{file_name}"
                    data = {'imageUrl': image_url}
                    api_url = st.secrets["api_url"]
                    
                    # Hacer la solicitud POST
                    response = requests.post(api_url, json=data)
                    
                    # Manejar la respuesta
                    try:
                        if response.status_code == 200:
                            result = response.json()
                            show_result = True
                        else:
                            st.write("Something went wrong")
                    except Exception as e:
                        st.write(f"Error: {e}")
        
    else:
        st.text("Please upload a cocktail menu image")

# Añadir textos en la segunda columna
with col2:
    st.header("Space for additional information")
    st.write("Aquí puedes poner algunos textos.")
    st.write("Este es un ejemplo de cómo puedes organizar el contenido en columnas.")
    st.write("¡Puedes añadir más detalles según lo necesites!")

if show_result:
    if result["menu"]["category"]=="cocktail":
        exclude_keys = ['category', 'information', 'execution_time', 'image']
        for key, value in result["menu"].items():
            if key not in exclude_keys:
                st.subheader(f"Cocktail: {key}")
                col1, colsep, col2, col3, col4 = st.columns([3, 0.1, 1, 1, 1])
                with col1:
                    ingredients = ", ".join(value)
                    st.markdown(f"**Ingredients:** {ingredients}")
    else:
        st.text("Please upload a cocktail menu image")