from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.sklearn
from sentence_transformers import SentenceTransformer
import json


# Charger un modèle de Sentence Transformers - embeddings plus riches
model = SentenceTransformer('/app/models/all-MiniLM-L6-v2')


# Fonction de nettoyage
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    tokens = text.split()  # Tokeniser
    tokens = [word for word in tokens if word not in stop_words]  # Supprimer les stopwords
    return " ".join(tokens)

def get_cluster_info_from_mongo(collection):
    """
    Récupère les embeddings et les thèmes depuis MongoDB, regroupe par thème et calcule les centroïdes.

    Paramètres :
    - collection : Collection MongoDB contenant les données.

    Retourne :
    - cluster_centroids (dict) : Centroïdes des clusters sous forme de dictionnaire {thème: centroid_embedding}.
    - cluster_themes (dict) : Dictionnaire associant chaque cluster à un thème {cluster_id: thème}.
    """

    # 1️⃣ Récupération des données depuis MongoDB
    data = list(collection.find({}, {"_id": 0, "theme": 1, "embedding": 1}))

    # 2️⃣ Vérification des données
    if not data:
        raise ValueError("Aucune donnée trouvée dans la base MongoDB ! 🚨")

    # 3️⃣ Regrouper les embeddings par thème
    theme_embeddings = {}
    for entry in data:
        theme = entry["theme"]
        embedding = entry["embedding"]

        if theme not in theme_embeddings:
            theme_embeddings[theme] = []
        theme_embeddings[theme].append(embedding)

    # 4️⃣ Calcul des centroïdes des clusters
    cluster_centroids = {theme: np.mean(embeddings, axis=0) for theme, embeddings in theme_embeddings.items()}

    # 5️⃣ Associer un ID numérique à chaque thème pour le mapping
    cluster_themes = {idx: theme for idx, theme in enumerate(theme_embeddings.keys())}

    return cluster_centroids, cluster_themes

def classify_new_data(new_data, model, cluster_centroids, cluster_themes):
    """
    Classifie une nouvelle donnée en utilisant les centroïdes des clusters récupérés depuis MongoDB.

    Paramètres :
    - new_data (dict) : Contient "Nom du dépôt", "Topics", "Description", "README".
    - model (SentenceTransformer) : Modèle SentenceTransformer pour générer l'embedding.
    - cluster_centroids (dict) : Centroïdes des clusters {thème: centroid_embedding}.
    - cluster_themes (dict) : Association entre ID de cluster et thème {cluster_id: thème}.

    Retourne :
    - theme (str) : Thème prédit pour la nouvelle donnée.
    - embedding (list) : L'embedding de la nouvelle donnée.
    """


    if isinstance(new_data, list):
        for repo in new_data:
            combined_text = (
                repo["Nom du dépôt"] + " " +
                (" ".join(repo["Topics"]) if isinstance(repo["Topics"], list) else "") + " " +
                repo["Description"] + " " +
                repo["README"]
            )

    else:
    # 1️⃣ Concaténer et nettoyer le texte
        combined_text = (
            new_data["Nom du dépôt"] + " " +
            (" ".join(new_data["Topics"]) if isinstance(new_data["Topics"], list) else "") + " " +
            new_data["Description"] + " " +
            new_data["README"]
        )
    cleaned_text = clean_text(combined_text)

    # 2️⃣ Générer l'embedding
    embedding = model.encode([cleaned_text])[0]

    # 3️⃣ Comparer avec les centroïdes des clusters
    themes = list(cluster_centroids.keys())  # Liste des thèmes
    centroids = np.array([cluster_centroids[theme] for theme in themes])

    similarities = cosine_similarity([embedding], centroids)[0]
    best_theme = themes[np.argmax(similarities)]  # Trouver le thème avec la meilleure similarité

    return best_theme, embedding.tolist()


# Fonction de traitement et de mise à jour de MongoDB
def kmeans_clustering_and_update_mongodb():
    # Connexion à MongoDB
    client = MongoClient("mongodb://mongodb:27017")
    db = client["my_database"]
    collection = db["my_collection"]

    # Charger les données
    data = list(collection.find({}, {"_id": 1, "Nom du dépôt": 1, "Topics": 1, "Description": 1, "README": 1, "embedding": 1}))
    
    # Convertir en DataFrame Pandas
    df = pd.DataFrame(data)
    
    # Combiner les textes
    df["combined_text"] = df["Nom du dépôt"] + " " + df["Topics"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " + df["Description"] + " " + df["README"]

    # Télécharger les stopwords de nltk
    nltk.download('stopwords')

    # Appliquer le nettoyage
    df["cleaned_text"] = df["combined_text"].apply(clean_text)

    # Utiliser les embeddings existants (pas de recalcul)
    # Assurez-vous que les embeddings sont sous forme de listes
    df["embedding"] = df["embedding"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)  # Assurez-vous que les embeddings sont en liste

    mlflow.set_tracking_uri("http://mlflow:5000")

    # Initialiser MLflow
    mlflow.set_experiment("KMeans Clustering Experiment")

    os.environ["OMP_NUM_THREADS"] = "1"  # Éviter les erreurs de leak de mémoire

    # Déterminer le nombre optimal de clusters avec l'analyse de silhouette
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

        # Sauvegarde du score de silhouette pour k optimal
        silhouette_collection = db["silhouette_scores"]  # Nouvelle collection pour stocker les scores
        silhouette_data = {
            "best_silhouette_score": float(best_silhouette_score),
            "run_id": run_id,
            "optimal_k": optimal_k
        }

        silhouette_collection.insert_one(silhouette_data)

        print(f"Le nombre optimal de clusters selon l'analyse de silhouette est : {optimal_k}")

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
        query = {"_id": row["_id"]}  # Assure-toi que cette clé est unique

        update_data = {
            "$set": {
                "embedding": row["embedding"],  # Conserver l'embedding déjà calculé
                "theme": cluster_themes[row["cluster"]]  # Associer le thème détecté
            }
        }
        
        result = collection.update_one(query, update_data, upsert=True)  # Met à jour ou insère si inexistant
        #print(f"Mise à jour du dépôt {row['Nom du dépôt']}: Matched {result.matched_count}, Modified {result.modified_count}")

    print("Mise à jour MongoDB terminée !")



def run(new_repo):
    """
    1️⃣ Classifie une nouvelle donnée avec les clusters existants.
    2️⃣ Récupère l'ancien k_optimal et silhouette_score depuis MongoDB.
    3️⃣ Recalcule le score de silhouette avec la nouvelle donnée.
    4️⃣ Compare les scores et met à jour si nécessaire.
    """
    # Conversion de la chaîne JSON en dictionnaire
    new_repo = json.loads(new_repo)

    # Afficher les données
    #print(new_repo)

    # Afficher les entrées du dictionnaire
    for repo in new_repo:
        print(f"Nom du dépôt: {repo['Nom du dépôt']}")
        print(f"Propriétaire: {repo['Propriétaire']}")
        print(f"Date de création: {repo['Date de création']}")
        print(f"URL: {repo['Url']}")
        print(f"Langage principal: {repo['Langage principal']}")
        print(f"Description: {repo['Description']}")
        print(f"README: {repo['README']}")
        print(f"Date de sauvegarde: {repo['Date de sauvegarde']}")
        print("------")


    # 🔹 Connexion MongoDB
    client = MongoClient("mongodb://mongodb:27017")
    db = client["my_database"]
    collection = db["my_collection"]
    silhouette_collection = db["silhouette_scores"]

    if isinstance(new_repo, list) and len(new_repo) == 1:
        new_repo = new_repo[0]

    # 🔹 Récupérer les centroïdes et les thèmes des clusters
    cluster_centroids, cluster_themes = get_cluster_info_from_mongo(collection)

    # 🔹 Classifier la nouvelle donnée
    predicted_theme, new_embedding = classify_new_data(new_repo, model, cluster_centroids, cluster_themes)

    print(f"🔹 Nouvelle donnée classée sous le thème : {predicted_theme}")

    # 🔹 Insérer la nouvelle donnée dans MongoDB
    collection.insert_one({
        **new_repo,
        "embedding": new_embedding,
        "theme": predicted_theme
    })
    print("✅ Nouvelle donnée insérée dans MongoDB !")

    # 🔹 Récupérer l'ancien k_optimal et l'ancien score silhouette depuis MongoDB
    silhouette_data = silhouette_collection.find_one({}, {"_id": 0, "best_silhouette_score": 1, "optimal_k": 1})
    
    if not silhouette_data:
        print("⚠️ Aucun score silhouette trouvé, recalcul nécessaire !")
        kmeans_clustering_and_update_mongodb()
        return
    
    old_silhouette_score = silhouette_data["best_silhouette_score"]
    optimal_k = silhouette_data["optimal_k"]

    print(f"📊 Ancien score silhouette : {old_silhouette_score}")
    print(f"📌 k_optimal stocké en base : {optimal_k}")

    # 🔹 Recalculer le score silhouette avec la nouvelle donnée
    df = pd.DataFrame(list(collection.find({}, {"_id": 0, "embedding": 1})))

    if df.shape[0] < optimal_k:
        print("⚠️ Pas assez de données pour recalculer le score silhouette. Recalcul des clusters nécessaire.")
        kmeans_clustering_and_update_mongodb()
        return
    
    # Convertir les embeddings en liste de vecteurs
    df["embedding"] = df["embedding"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    embeddings = df["embedding"].tolist()

    # Appliquer K-Means avec k_optimal
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    new_silhouette_score = silhouette_score(embeddings, labels)

    print(f"📊 Nouveau score silhouette : {new_silhouette_score}")

    # 🔹 Comparer et décider de mettre à jour ou non
    if new_silhouette_score > old_silhouette_score*1.15:
        print("✅ Nouveau score meilleur, recalcul des clusters !")
        kmeans_clustering_and_update_mongodb()
    else:
        print("🔹 Pas de recalcul nécessaire, conservation des clusters existants.")