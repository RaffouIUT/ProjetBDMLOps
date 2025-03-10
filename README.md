### Description du Projet
Ce projet vise à faire un clustering de dépôts Github par thématiques en utilisant les principes de Big Data et MLOps.
Utilisation des README, langages, topics et descriptions des différents dépôts.

### Installation
1. Clonez le dépôt : git clone https://github.com/RaffouIUT/ProjetBDMLOps.git

2. Allez dans le dossier : cd ProjetBDMLOps

3. Ajoutez un fichier .env avec la clé Github dedans sous la forme : CLE_GITHUB="VOTRE_CLE_GITHUB"

3. Installez les applications Docker et MongoDB Compass
   
4. Lancez docker-compose up --build -d

5. Dans l'application MongoDB Compass, rajoutez une base avec comme nom my_database et une collection avec comme nom my_collection et ajoutez le fichier JSON présent dans le dossier

### Utilisation
1. Lancez au début le fichier launch.py : docker exec -it fastapi-app python launch.py

2. Lancez le site avec l'adresse : localhost:8000

3. Lancez la commande dans un terminal : docker exec -it fastapi-app python consumer.py

4. Ajoutez /add_random_data dans l'adresse afin de rechercher un nouveau dépôt et l'ajouter dans la base

### Potentiels problèmes

Si lors du lancement de la commande docker-compose up --build -d, vous remarquez que le container Kafka se ferme quelques secondes après le lancement, supprimer le container ET le volume du même nom (cela devrait résoudre le problème de persistence des données).

### Contributeurs
Rafaël Doneau - Léo Notelet