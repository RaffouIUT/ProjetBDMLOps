from requests.auth import HTTPBasicAuth
from fastapi import FastAPI
from elasticsearch import Elasticsearch
import nest_asyncio
import base64
import random
import csv
from markdown import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from github import Github
from pymongo import MongoClient
import datetime
from producer import send_message




# Charger le fichier .env
load_dotenv()

# Accéder aux variables d'environnement
api_key = os.getenv("CLE_GITHUB")

# Create FastAPI app
app = FastAPI()

# Connect to Elasticsearch using container name
username = "elastic"
password = "password"
es = Elasticsearch("http://elasticsearch:9200", http_auth=(username, password))

def serialize_doc(doc):
    """Convertit _id de MongoDB en string pour JSON."""
    doc['_id'] = str(doc['_id'])
    return doc

@app.get("/")
async def read_root():
    return {"message": "Welcome to the API GitHub data finder"}


@app.get("/mongodb_db")
async def get_data():
    client = MongoClient("mongodb:27017")
    db = client["my_database"]
    collection = db["my_collection"]
    data = list(collection.find())
    item_list = [serialize_doc(item) for item in data]
    return {"message": "test mongosh","data":item_list}



@app.get("/add_random_data")
async def add_data():

    # Remplacez par votre token d'accès personnel
    TOKEN = api_key

    # Connexion à l'API GitHub
    g = Github(TOKEN)

    # Connexion à MongoDB
    client = MongoClient("mongodb://mongodb:27017")
    db = client["my_database"]
    collection = db["my_collection"]

    # Fonction pour récupérer des dépôts aléatoires
    def get_random_repositories(github, num_repos=1):
        """Récupère un certain nombre de dépôts publics aléatoires distincts."""
        repositories = []
        seen_repos = set()  # Ensemble pour éviter les doublons

        data = list(collection.find())
        for item in data:
            seen_repos.add(item["Propriétaire"]+"/"+item["Nom du dépôt"])

        search_query = "stars:>1"  # Rechercher des dépôts avec au moins 1 étoile

        for repo in github.search_repositories(query=search_query):
            # Utiliser un identifiant unique pour vérifier les doublons (par exemple, le nom complet)
            repo_identifier = repo.full_name  # Exemple : "owner/repository_name"

            if repo_identifier not in seen_repos:
                repositories.append(repo)
                seen_repos.add(repo_identifier)

            # Arrêter la boucle si on a récupéré suffisamment de dépôts
            if len(repositories) >= num_repos:
                break

        return repositories

    # Fonction pour convertir le Markdown en texte brut
    def markdown_to_text(markdown_content):
        # Convertir le Markdown en HTML
        html_content = markdown(markdown_content)
        # Extraire le texte brut du HTML
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    # Fonction pour supprimer les sauts de ligne multiples
    def remove_multiple_newlines(text):
        # Remplace plusieurs sauts de ligne consécutifs par un seul saut de ligne
        return "\n".join(line for line in text.splitlines() if line.strip())

    # Fonction pour extraire les informations des dépôts
    def extract_repo_data(repo):
        """Extrait les informations d'un dépôt donné."""
        try:
            readme = repo.get_readme()
            readme_content = base64.b64decode(readme.content).decode("utf-8")
        except:
            readme_content = "README non disponible"

        last_commit = repo.get_commits()[0]    
        return {
            "Nom du dépôt": repo.name,
            "Propriétaire": repo.owner.login,
            "Topics":repo.get_topics(),
            "Date de création": repo.created_at,
            "Url": repo.html_url,
            "Date du dernier commit": last_commit.commit.author.date if last_commit else "N/A",
            "Langage principal": repo.language,
            "Nombre d'étoiles": repo.stargazers_count,
            "Description": repo.description or "Pas de description",
            "README": remove_multiple_newlines(markdown_to_text(readme_content))  # Limiter à 1000 caractères
        }

    # Récupérer des dépôts aléatoires
    num_repos = 1  # Nombre de dépôts à collecter
    random_repositories = get_random_repositories(g, num_repos)

    # Extraire les informations de chaque dépôt
    data = []
    i = 0
    for repo in random_repositories:
        repo_data = extract_repo_data(repo)
        data.append(repo_data)
        i+=1
        print(f"Dépôt analysé {i}: {repo_data['Nom du dépôt']}")

    # Ajouter la date de sauvegarde
    for repo in data:
        repo["Date de sauvegarde"] = datetime.datetime.now()

    # Insérer les données dans la collection
    #collection.insert_many(data)
    item_list = list(data)
    #item_list = [serialize_doc(item) for item in data_list]

    send_message(item_list)

    return{"message":"Données des dépôts sauvegardées sur mongodb","data":item_list}
