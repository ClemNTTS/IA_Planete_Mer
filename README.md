# ChatBot Planète Mer 🐠

Un chatbot utilisant Milvus pour le stockage de vecteurs et LangChain pour le traitement du langage naturel.

## Prérequis

### Installation de Docker
- **Pour MacOS et Windows**
  - Installer [Docker Desktop](https://www.docker.com/products/docker-desktop)
  
- **Pour Linux**
  - Installer Docker Engine
  - Installer Docker Compose

### Dépendances Python
```bash
pip install streamlit langchain-community langchain-huggingface pymilvus pdf2image pytesseract PyPDF2 tqdm
```

## Installation

1. Créer le dossier du projet
```bash
mkdir mon_chatbot
cd mon_chatbot
```

2. Cloner les fichiers du projet
```bash
git clone <votre-repo>
```

3. Lancer Milvus avec Docker Compose
```bash
docker compose up -d
```

4. Vérifier que les conteneurs sont en cours d'exécution
```bash
docker ps
```

Vous devriez voir trois conteneurs :
- milvus
- etcd
- minio

## Structure du Projet

```
mon_chatbot/
├── docker-compose.yml   # Configuration Docker
├── chat.py              # Interface Streamlit
├── indexer.py           # Traitement et indexation des documents
├── data/                # Dossier pour les documents PDF
└── volumes/             # Volumes Docker pour Milvus
```

## Utilisation
1. Activate venv
```bash
source venv/bin/activate
```
2. Indexation des documents
```bash
python indexer.py
```
3. Lancer le chatbot
```bash
streamlit run chat.py
```

Le chatbot sera accessible à l'adresse : `http://localhost:8501`

## Fonctionnalités

- Interface utilisateur conviviale avec Streamlit
- Traitement de documents PDF avec support OCR
- Stockage vectoriel avec Milvus
- Réponses basées sur le contexte avec LangChain
- Historique des conversations
- Affichage des sources des réponses

## Configuration

### Modification des paramètres de l'indexeur
Dans `indexer.py`, vous pouvez ajuster :
- Taille des chunks (`chunk_size`)
- Chevauchement des chunks (`chunk_overlap`)
- Modèle d'embedding (`model_name`)
- Nom de la collection Milvus (`collection_name`)

### Configuration du chatbot
Dans `chat.py`, vous pouvez personnaliser :
- Le modèle LLM
- Le nombre de documents pertinents à récupérer
- Le style de l'interface utilisateur

## Dépannage

1. Si Milvus ne démarre pas :
```bash
docker compose down
docker compose up -d
```

2. Pour réinitialiser les données :
```bash
docker compose down -v
docker compose up -d
```