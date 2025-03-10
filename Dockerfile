# Utilisation de l'image officielle Python
FROM python:3.10

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application
COPY requirements.txt requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Installer les dépendances NLTK
RUN python -m nltk.downloader stopwords

# Copier les fichiers de l'application
COPY . .

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Démarrer l'application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
