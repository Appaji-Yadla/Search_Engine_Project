import re
import os
import json
import chromadb
from text_preprocessing import clean_text
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# Specify the path to the folder containing the config file
config_folder_path = "Subtitle_SemanticSearch_files"

# Load the config file
config_file_path = os.path.join(config_folder_path, "config_sentence_transformers.json")

# Check if the file exists
if os.path.exists(config_file_path):
    # Load the config file
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)
else:
    print("Config file not found.")

# Load pre-trained embedding model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Load ChromaDB collection
client = chromadb.PersistentClient(path='ChromaDB')
collection = client.get_collection(name="mydata__collection")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_query  = request.form.get("text")
        # Check if the user entered text
        if not user_query:
            return render_template('home.html', output=["Please enter some text"])
        
        # Preprocess the text using the preprocessor function
        processed_text = clean_text(user_query)
        
        query_embedding  = model.encode(processed_text).tolist()

        # Retrieve similar documents from ChromaDB based on cosine similarity
        result = collection.query(
               query_embeddings=[query_embedding],
               n_results=5,
               include=["metadatas"]
          )
               
        # Extract the metadata names of similar documents
        similar_names = []
        for item in result['metadatas']:
            for metadata in item:
                similar_names.append(metadata['name'])
                 
        return render_template('home.html', output=similar_names)
    else:
        # Redirect to home page if accessed directly
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
