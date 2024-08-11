import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from pinecone import Pinecone
import os
import json
import requests
from contextlib import asynccontextmanager
from typing import Optional

load_dotenv()
PINECONE_KEY = os.getenv("PINECONE_KEY")
hf_token = os.getenv("HF_TOKEN")
model_id = "sentence-transformers/all-MiniLM-L6-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

pc = Pinecone(
    api_key=PINECONE_KEY,
)
index_name = 'semantic-search'
index = pc.Index(index_name)

def search_query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

app = FastAPI()

@app.get('/')
def semantic_search(q:Optional[str] = None):
    query = q or "minimum resource quantum computation increase"
    input = [query]
    output = search_query(input)
    res = index.query(vector = output, top_k=5, include_metadata=True)
    result = {}
    for i in range(0, 5):
        temp_dict = {}
        for key in res['matches'][i].to_dict():
            if key != 'values':
                temp_dict[key] = res['matches'][i].to_dict()[key]
        result[i + 1] = temp_dict
    return json.dumps(result)





