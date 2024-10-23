from openai import OpenAI
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from PIL import Image
import io
from io import StringIO
import boto3
import requests

@st.cache_data
def load_data_from_s3(object_name):
    client = boto3.client('s3', 
                          aws_access_key_id=st.secrets['aws_access_key_id'], 
                          aws_secret_access_key=st.secrets['aws_secret_access_key'])
    bucket_name = st.secrets["bucket_mixo_data"]
    
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_name)
    body = csv_obj['Body'].read().decode('utf-8')

    return pd.read_csv(StringIO(body))

object_name = "cocktails_extended.csv"
df_cocktails_info = load_data_from_s3(object_name)

n_cocktails = df_cocktails_info.shape[0]


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

def get_embeddings(texts, model="text-embedding-3-small"):
    texts = texts.split(", ")
    embeddings = client.embeddings.create(input=texts, model=model).data
    mean_embedding = np.array([embedding.embedding for embedding in embeddings]).mean(axis=0)
    mean_embedding = mean_embedding.reshape(1, len(mean_embedding))
    return mean_embedding

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
    st.write(f"Total cocktails: {n_cocktails}")
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