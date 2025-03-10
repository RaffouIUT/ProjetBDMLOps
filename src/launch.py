# Lancer le script de clustering à l'initialisation
# python src/launch.py



from pymongo import MongoClient
import pandas as pd

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

from sentence_transformers import SentenceTransformer

import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

import numpy as np

# Connexion à MongoDB
client = MongoClient("mongodb://mongodb:27017")
db = client["my_database"]
collection = db["my_collection"]

# Charger les données
data = list(collection.find({}, {"_id": 0, "Nom du dépôt": 1, "Topics": 1, "Description": 1, "README": 1}))

# Convertir en DataFrame Pandas
df = pd.DataFrame(data)
df["combined_text"] = df["Nom du dépôt"] + " " + df["Topics"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " + df["Description"] + " " + df["README"]

# Télécharger les stopwords de nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Fonction de nettoyage
def clean_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    tokens = text.split()  # Tokeniser
    tokens = [word for word in tokens if word not in stop_words]  # Supprimer les stopwords
    return " ".join(tokens)

# Appliquer le nettoyage
df["cleaned_text"] = df["combined_text"].apply(clean_text)

# Charger un modèle de Sentence Transformers - embeddings plus riches
model = SentenceTransformer('/app/models/all-MiniLM-L6-v2')

# Convertir les textes en embeddings
df["embedding"] = list(model.encode(df["cleaned_text"].tolist()))

mlflow.set_tracking_uri("http://localhost:5000")

# Initialiser MLflow
mlflow.set_experiment("KMeans Clustering Experiment")

os.environ["OMP_NUM_THREADS"] = "1"  # Éviter les erreurs de leak de mémoire

# Déterminer le nombre optimal de clusters avec la silhouette analysis
silhouette_scores = []
inertia = []
K = range(2, min(10, df.shape[0] + 1))  # Limiter K au nombre d'échantillons

with mlflow.start_run() as run:
    run_id = run.info.run_id  # Récupérer l'ID de la run
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df["embedding"].tolist())
        score = silhouette_score(df["embedding"].tolist(), labels)
        
        silhouette_scores.append(score)
        inertia.append(kmeans.inertia_)

        # Enregistrer les métriques dans MLflow
        mlflow.log_metric(f"silhouette_score_k{k}", score)
        mlflow.log_metric(f"inertia_k{k}", kmeans.inertia_)

    # Tracer la courbe d'inertie et l'enregistrer
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, marker='o')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour choisir k')
    plt.savefig("mlruns/elbow_method.png")
    mlflow.log_artifact("mlruns/elbow_method.png")  # Enregistrer l’image dans MLflow
    plt.close()

    # Tracer le score de silhouette et l'enregistrer
    plt.figure(figsize=(8, 5))
    plt.plot(K, silhouette_scores, marker='o')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Score de silhouette')
    plt.title('Analyse de silhouette pour choisir k')
    plt.savefig("mlruns/silhouette_analysis.png")
    mlflow.log_artifact("mlruns/silhouette_analysis.png")  # Enregistrer l’image dans MLflow
    plt.close()

    # Trouver le meilleur score de silhouette
    best_silhouette_score = max(silhouette_scores)

    # Trouver le k optimal
    optimal_k = K[silhouette_scores.index(best_silhouette_score)]
    mlflow.log_param("optimal_k", optimal_k)  # Enregistrer k optimal
    print(f"Le nombre optimal de clusters selon l'analyse de silhouette est : {optimal_k}")

    # Sauvegarde du score de silhouette pour k optimal
    silhouette_collection = db["silhouette_scores"]  # Nouvelle collection pour stocker les scores
    silhouette_data = {
        "best_silhouette_score": float(best_silhouette_score),
        "run_id": run_id,
        "optimal_k": optimal_k
    }

    silhouette_collection.insert_one(silhouette_data)

    # Appliquer K-Means avec le k optimal
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df["cluster"] = kmeans.fit_predict(df["embedding"].tolist())

    # Enregistrer le modèle K-Means dans MLflow
    mlflow.sklearn.log_model(kmeans, "KMeans_Model")

    print("Modèle K-Means enregistré avec MLflow !")

    # Regrouper les textes par cluster
clusters = df["cluster"].unique()
cluster_texts = {cluster: df[df["cluster"] == cluster]["cleaned_text"].tolist() for cluster in clusters}
cluster_themes = {}
cluster_centroids = {cluster: np.mean(df[df["cluster"] == cluster]["embedding"].tolist(), axis=0) for cluster in clusters}

for cluster, texts in cluster_texts.items():
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2, ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    top_indices = np.argsort(tfidf_scores)[::-1][:10]
    top_keywords = [feature_names[i] for i in top_indices]
    
    if not top_keywords:
        cluster_themes[cluster] = "Unknown"
        continue
    
    keyword_embeddings = model.encode(top_keywords)
    centroid = cluster_centroids[cluster]
    similarities = np.dot(keyword_embeddings, centroid)
    
    best_keyword_index = np.argmax(similarities)
    best_keyword = top_keywords[best_keyword_index]
    
    other_centroids = [cluster_centroids[c] for c in clusters if c != cluster]
    other_similarities = [np.dot(model.encode(best_keyword), c) for c in other_centroids]
    
    if max(other_similarities) > 0.1:
        similarities[best_keyword_index] = -1
        best_keyword_index = np.argmax(similarities)
        best_keyword = top_keywords[best_keyword_index]
    
    cluster_themes[cluster] = best_keyword

print("Thèmes des clusters :")
for cluster, theme in cluster_themes.items():
    print(f"Cluster {cluster}: {theme}")

# Mettre à jour chaque document avec l'embedding et le thème
for _, row in df.iterrows():
    query = {"Nom du dépôt": row["Nom du dépôt"]}  # Assure-toi que cette clé est unique

    update_data = {
        "$set": {
            "embedding": row["embedding"].tolist(),  # Convertir en liste pour MongoDB
            "theme": cluster_themes[row["cluster"]]  # Associer le thème détecté
        }
    }
    
    collection.update_one(query, update_data, upsert=True)  # Met à jour ou insère si inexistant

print("Mise à jour MongoDB terminée !")
