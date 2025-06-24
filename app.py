"""
arXiv RAG Web Application

A Flask-based web application that implements Retrieval-Augmented Generation (RAG)
for querying computer science research papers from arXiv. Combines semantic search
using FAISS with AI-powered response generation via Google Gemini.

Author: Ahmed Mossad
Created: 2025
License: MIT
"""

import os
import numpy as np
import pandas as pd
import faiss
import re
import time
import torch
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import pickle
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global variables for loaded models and data
embedding_model = None
index = None
df_chunks = None
embedding_dim = None

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_text(text):
    """
    Enhanced text normalization function optimized for scientific/academic papers:
    - Preserves technical terminology while standardizing format
    - Retains important numerical values and model names
    - Handles scientific abbreviations and technical terms appropriately
    - Preserves sentence structure for technical papers
    """
    if not isinstance(text, str) or not text:
        return ""

    # Convert to lowercase (optional for scientific text)
    text = text.lower()

    # Remove URLs and email addresses
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' EMAIL ', text)

    # Standardize quotation marks and special characters
    text = re.sub(r'[""''`]', '"', text)
    text = re.sub(r'[–—−]', '-', text)

    # Replace newlines, tabs, and other whitespace characters with spaces
    text = re.sub(r'[\n\t\r\f\v]+', ' ', text)

    # Handle LaTeX-style math expressions but preserve them in a standardized form
    text = re.sub(r'\$+([^$]*)\$+', r' MATH_EXPRESSION(\1) ', text)

    # Replace multiple punctuation with single instance (e.g., !!! -> !)
    text = re.sub(r'([.,!?;:]){2,}', r'\1', text)

    # Add space around punctuation while preserving technical terms with dashes or periods
    text = re.sub(r'(?<!\w)([.,!?;:])(?!\w)|(?<=\w)([.,!?;:])(?!\w)|(?<!\w)([.,!?;:])(?=\w)', r' \1\2\3 ', text)

    # Preserve hyphenated terms and compounds common in technical writing
    text = re.sub(r'(?<=\w)-(?=\w)', r'-', text)

    # Preserve common technical abbreviations and model names
    # Add specific patterns for technical terms in your domain
    tech_terms = r'(?<!\w)(3d|2d|cnn|lstm|ai|ml|cv|nlp|gpu|cpu)(?!\w)'
    text = re.sub(tech_terms, lambda m: m.group(0).upper(), text)

    # Keep alphanumeric characters, spaces, and technical punctuation
    text = re.sub(r'[^\w\s.,;:?!"\'\-()]', ' ', text)

    # Handle scientific abbreviations and technical notations
    text = re.sub(r'\b(fig|eq|eqn|i\.e|e\.g|et al|vs|etc|cf|viz|resp|approx|w\.r\.t|a\.k\.a)\s*\.', r'\1.', text)

    # Handle numbered lists and sections often found in papers
    text = re.sub(r'(\d+)\.\s+', r'\1. ', text)

    # Collapse multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def retrieve(query, k=5):
    """
    Retrieve the top-k most similar chunks for a given query
    """
    global index, df_chunks, embedding_model
    
    # Normalize the query text
    query_normalized = normalize_text(query)

    # Embed the query
    query_embedding = embedding_model.encode([query_normalized], convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()

    # Normalize the query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search the index
    scores, indices = index.search(query_embedding, k)

    # Extract scores and indices from first row (batch size=1)
    scores = scores[0]
    indices = indices[0]

    # Get the retrieved chunks as a DataFrame
    chunks_df = df_chunks.iloc[indices].copy()

    # Add similarity scores
    chunks_df['similarity_score'] = scores

    return chunks_df

def construct_prompt(query, retrieved_chunks, k=3):
    """
    Construct a prompt for the LLM using the retrieved chunks
    """
    # Limit to top k chunks
    top_chunks = retrieved_chunks.head(k)

    # Create context section with chunks
    context_sections = []
    for i, (_, chunk) in enumerate(top_chunks.iterrows()):
        context_sections.append(f"[{i+1}] From paper: \"{chunk['title']}\"\n{chunk['chunk_text']}")

    context = "\n\n".join(context_sections)

    # Construct the prompt
    prompt = f"""Context:

    {context}

    Question: {query}

    Answer:"""

    return prompt

def generate_response(prompt, model_name="models/gemini-2.5-flash-preview-04-17"):
    """
    Send the prompt to the Google Gemini API.
    """
    try:
        # Set the API key from environment variable
        if os.environ.get("GOOGLE_API_KEY"):
            api_key = os.environ.get("GOOGLE_API_KEY")
        else:
            # Fallback to a default key if not set (not recommended for production)
            api_key = "YOUR_API_KEY"
            print("Warning: Using default API key. Set GOOGLE_API_KEY in .env file.")
        
        # Configure the API key for the Gemini client
        genai.configure(api_key=api_key)

        # Create the model
        model = genai.GenerativeModel(model_name)

        # Generate content
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.8,
                "max_output_tokens": 1024,
                "top_p": 0.95,
            }
        )

        return response.text

    except Exception as e:
        return f"Error generating response: {str(e)}"

def rag_pipeline(query, k=3):
    """
    Run the full RAG pipeline for a given query
    """
    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query, k=3)

    # Get top sources for display
    top_sources = [{"title": row["title"], "score": row["similarity_score"]} 
                 for _, row in retrieved_chunks.iterrows()]

    # Construct prompt
    prompt = construct_prompt(query, retrieved_chunks, k=3)

    # Generate response
    response = generate_response(prompt)

    return {
        "response": response,
        "sources": top_sources
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    k = int(data.get('k', 3))
    
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    try:
        result = rag_pipeline(query_text, k=3)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_models():
    """
    Load the necessary models and data for the RAG pipeline
    """
    global embedding_model, index, df_chunks, embedding_dim
    
    print("Loading saved data and models...")
    
    # Set up the base path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load data files
    data_path = os.path.join(base_path, 'arxiv_rag_data.pkl')
    index_path = os.path.join(base_path, 'arxiv_faiss_index')
    
    # Load data
    with open(data_path, 'rb') as f:
        saved_data = pickle.load(f)

    df_chunks = saved_data['chunks']
    embedding_dim = saved_data['embedding_dim']

    # Load embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

    # Load FAISS index
    index = faiss.read_index(index_path)
    
    print("Models and data loaded successfully")

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
