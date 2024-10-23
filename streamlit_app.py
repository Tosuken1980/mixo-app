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
import joblib
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
s3 = boto3.client('s3', aws_access_key_id=st.secrets['aws_access_key_id'], aws_secret_access_key=st.secrets['aws_secret_access_key'])
secret_token = st.secrets['token']
bucket = st.secrets["bucket_image_dwls"]
bucket_name = st.secrets["bucket_mixo_data"]
object_name = "cocktails_extended.csv"
pca_model_name = "pca_model.pkl"
prefix = "web-images/"


csv_obj = s3.get_object(Bucket=bucket_name, Key=object_name)
body = csv_obj['Body'].read().decode('utf-8')
df_cocktails_info = pd.read_csv(StringIO(body))

embeddings_data = df_cocktails_info.loc[:, df_cocktails_info.columns.str.startswith('PC')].to_numpy()
n_cocktails = df_cocktails_info.shape[0]

s3_response_object_pca = s3.get_object(Bucket=bucket_name, Key=pca_model_name)
file_stream = io.BytesIO(s3_response_object_pca['Body'].read())
pca = joblib.load(file_stream)

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
    embedding=np.array([embedding.embedding for embedding in embeddings])
    return pd.DataFrame(embedding)

def find_similarities(embedding_reference, embeddings_data):
    similarities = cosine_similarity(embeddings_data, embedding_reference)
    df_cocktails_info['similarity_v2v'] = similarities.flatten()
    df_sorted = df_cocktails_info.sort_values(by='similarity_v2v', ascending=False)
    filtered_recipes = df_sorted[df_sorted['similarity_v2v'] > 0.55]
    top_recipes = filtered_recipes if len(filtered_recipes) <= 15 else df_sorted.head(15)
    top_recipes = top_recipes[["cocktail_name", "transformed_ingredients", "cocktail_preparation", "cocktail_appearance", "temperature_serving", "similarity_v2v"]]

    return top_recipes

def estimate_cocktail_class(top_recipes,col):
    grouped_similarities = top_recipes.groupby(col)['similarity_v2v'].sum()
    total_similarity = grouped_similarities.sum()
    grouped_percentages = (grouped_similarities / total_similarity) * 100
    category = grouped_percentages.idxmax()
    probability = np.round(grouped_percentages[grouped_percentages.idxmax()],2)
    return category, probability


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
    st.write(f"Database cocktails: {n_cocktails}")
    #st.write("Este es un ejemplo de cómo puedes organizar el contenido en columnas.")
    #st.write("¡Puedes añadir más detalles según lo necesites!")

if show_result:
    if result["menu"]["category"]=="cocktail":
        exclude_keys = ['category', 'information', 'execution_time', 'image']
        for key, value in result["menu"].items():
            if key not in exclude_keys:
                st.subheader(f"Cocktail: {key}")
                col1, colsep, col2 = st.columns([3, 0.1, 2])
                with col1:
                    ingredients = ", ".join(value)
                    st.markdown(f"**Ingredients:** {ingredients}")
                with col2:
                    ingredients = ", ".join(value)
                    embeddings = get_embeddings(ingredients, model="text-embedding-3-small")
                    embeddings_pca = pca.transform(embeddings).mean(axis=0)
                    embeddings_pca = embeddings_pca.reshape(1, len(embeddings_pca))
                    top_recipes = find_similarities(embeddings_pca, embeddings_data)
                    cocktail_info = {}
                    if top_recipes.shape[0]>=1:
                        category_preparation, probability_preparation = estimate_cocktail_class(top_recipes,"cocktail_preparation")
                        category_appearance, probability_appearance = estimate_cocktail_class(top_recipes,"cocktail_appearance")
                        category_temperature, probability_temperature = estimate_cocktail_class(top_recipes,"temperature_serving")
                        cocktail_info = {
                            'cocktail_appearance': [category_appearance, probability_appearance],
                            'temperature_serving': [category_temperature, probability_temperature],
                            'cocktail_preparation': [category_preparation, probability_preparation],
                        }
                        st.subheader("This cocktail should be:")
                        for key, value in cocktail_info.items():
                            st.write(f"- **{value[0]}:** {value[1]}")
                    else:
                        cocktail_info = {
                            'total_cocktails': 0
                        } 
                        st.markdown(f"We do not have enough information for evaluate this cocktail")

    else:
        st.text("Please upload a cocktail menu image")