from pymongo import MongoClient
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Connexion à MongoDB
client = MongoClient("mongodb://127.0.0.1:27017")
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

#Avec TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limite les dimensions
X1 = vectorizer.fit_transform(df["cleaned_text"]).toarray()

#Avec sentence-transformers
from sentence_transformers import SentenceTransformer

# Charger un modèle de Sentence Transformers - embeddings plus riches
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convertir les textes en embeddings
X = model.encode(df["cleaned_text"].tolist())

os.environ["OMP_NUM_THREADS"] = "1"  # Pour éviter les erreurs de leak de mémoire sur Windows

# Déterminer le nombre optimal de clusters avec la silhouette analysis
silhouette_scores = []
inertia = []
K = range(2, min(10, X.shape[0] + 1))  # Limiter K au nombre d'échantillons

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    inertia.append(kmeans.inertia_)


# Tracer la courbe d'inertie
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour choisir k')
plt.savefig('inertie_plot.png')  # Sauvegarder l'image
plt.close()  # Fermer la figure pour éviter d'afficher dans l'interface

# Tracer le score de silhouette pour chaque k
plt.figure(figsize=(8, 5))
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Score de silhouette')
plt.title('Analyse de silhouette pour choisir k')
plt.savefig('silhouette_plot.png')  # Sauvegarder l'image
plt.close()

# Trouver le k optimal
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]

#test à supprimer avec k=4
optimal_k = 4

print(f"Le nombre optimal de clusters selon l'analyse de silhouette est : {optimal_k}")

# Appliquer K-Means avec le k optimal
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Afficher les données regroupées par cluster
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(df[df["cluster"] == cluster][["Nom du dépôt"]])
    
# Vérifier la taille des données
n_samples = X.shape[0]
perplexity = min(30, n_samples - 1)  # La perplexité doit être inférieure au nombre d'échantillons

# Réduction de dimensions
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
X_2d = tsne.fit_transform(X)

# Tracer les clusters
plt.figure(figsize=(10, 7))
for cluster in range(optimal_k):
    plt.scatter(X_2d[df["cluster"] == cluster, 0], X_2d[df["cluster"] == cluster, 1], label=f"Cluster {cluster}")

plt.legend()
plt.title("Clusters K-Means sur les données textuelles")
plt.savefig('Clusters K-Means.png')  # Sauvegarder l'image
plt.close()

# Réduction de dimensions avec PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Tracer les clusters
plt.figure(figsize=(10, 7))
for cluster in range(optimal_k):
    plt.scatter(X_2d[df["cluster"] == cluster, 0], X_2d[df["cluster"] == cluster, 1], label=f"Cluster {cluster}")

plt.legend()
plt.title("Clusters K-Means avec PCA")
plt.savefig('Clusters K-Means avec PCA.png')  # Sauvegarder l'image
plt.close()

# Exemple : Liste des vecteurs d'embedding et des clusters
# X est une matrice d'embedding où chaque ligne correspond à un vecteur
# df['cluster'] contient les étiquettes des clusters
clusters = df["cluster"].unique()
cluster_embeddings = {cluster: X[df["cluster"] == cluster] for cluster in clusters}

# Calculer le vecteur central pour chaque cluster
cluster_centroids = {
    cluster: np.mean(embeddings, axis=0)
    for cluster, embeddings in cluster_embeddings.items()
}

# Charger un modèle Sentence Transformers
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Liste de mots de vocabulaire potentiellement pertinents (vous pouvez personnaliser)
vocabulary = ["programming", "API", "machine learning", "JavaScript", "Python", "database", "web"]

# Embeddings des mots du vocabulaire
vocabulary_embeddings = sentence_model.encode(vocabulary)

# Trouver le mot le plus proche pour chaque cluster
cluster_themes = {}
for cluster, centroid in cluster_centroids.items():
    similarities = np.dot(vocabulary_embeddings, centroid)  # Produit scalaire
    best_match_index = np.argmax(similarities)  # Indice du mot le plus proche
    cluster_themes[cluster] = vocabulary[best_match_index]

print("Thèmes des clusters :")
for cluster, theme in cluster_themes.items():
    print(f"Cluster {cluster}: {theme}")