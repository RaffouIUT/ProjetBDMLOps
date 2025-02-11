from gridPower import logger
from requests.auth import HTTPBasicAuth
from fastapi import FastAPI
from elasticsearch import Elasticsearch
import nest_asyncio

logger.info("Welcome to MLOps courses")

# Create FastAPI app
app = FastAPI()

# Connect to Elasticsearch using container name
username = "elastic"
password = "password"
es = Elasticsearch("http://elasticsearch:9200", http_auth=(username, password))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the API GitHub data finder"}

# Do NOT include `uvicorn.run()` here! It will be run via `docker-compose`
