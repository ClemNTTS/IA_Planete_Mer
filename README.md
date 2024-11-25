# Prémis 

Veuillez clone ce projet :

- Copiez le lien :  https://github.com/ClemNTTS/IA_Planete_Mer.git

- Ouvrez VSCODE

- Clonnez le lien : https://github.com/ClemNTTS/IA_Planete_Mer.git


## Installation

### Python

Vérifier si `pyhton` et `pip` sont installés sur votre machine

```
python --version
```

sinon pour installer python: https://www.python.org/downloads/

Si `pip` n'est pas installé, vous pouvez l'installer avec :

```
python -m ensurepip --upgrade
```

### Ollama (modèle d'IA)

- Pour télécharger ollama : https://ollama.com/download

- Ouvrez votre terminal et entrez : `ollama pull llama3.2`


## Création d'environnement python

1) Créer un environnement python

```
python -m venv planetmer
```

Activer l'environment python sur windows

```
.\env\Scripts\activate
```

sur mac/linux :

```
source planetmer/bin/activate
```

2) Installer toutes les dépendances

Pour que votre projet fonctionne correctement vous devez installer les dépendances. 

```
pip install -r requirements.txt
```

`requirements.txt` est le fichier ou se trouve toutes les dépendances/package

## Indexation des pdf/data

Pour que le model puisse s'appuyer sur les données, il faut indexer les pdf dans une base de donnée.

Pour résumer : 

1) Lire les fichiers 
2) Séparer les textes en portions
3) `Embedding` : Configurer un modèle de représentation de texte, appelé embedding model, pour transformer du texte en un format que les ordinateurs peuvent facilement comprendre : des vecteurs (des listes de nombres).

#### À quoi ça sert ?
Transformer du texte en vecteurs numériques :

> Une phrase comme "J'adore les chats" devient une liste de nombres comme [0.12, -0.43, 0.56, ...].
Faire des comparaisons entre textes :

Grâce aux vecteurs, on peut mesurer à quel point deux phrases sont proches en calculant la distance entre leurs vecteurs (plus la distance est petite, plus elles se ressemblent).
Applications concrètes :

Recherche sémantique : Trouver des documents ou des réponses pertinentes dans une base de données.
Clustering : Regrouper des textes similaires.
Classification : Savoir de quoi parle un texte (exemple : spam ou pas spam).
Résumé de texte ou traduction.

4) Création de la base de donnée 

## Lancement du projet

Pour lancer le programme 

```
streamlit run chat.py
```

## Configuration du modèle

```python
db = Chroma(persist_directory="./db-planet-mer", embedding_function=embedding_model)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.7})
```

`Chroma` : Une base de données conçue pour stocker des documents (textes) sous forme de vecteurs numériques.

`Retrieval` : Quand l'utilisateur pose une question, on recherche les documents les plus pertinents en fonction de leur proximité avec la question (mesurée grâce aux vecteurs).

`Paramètres` :
k=5 : On récupère les 5 documents les plus pertinents.
lambda_mult=0.7 : On équilibre entre pertinence et diversité des résultats.


```python
llm = ChatOllama(model="llama3.2", keep_alive="3h", max_tokens=512, temperature=0.3)
```

`Modèle llama3.2`: C'est une intelligence artificielle spécialisée dans le dialogue. Elle génère des réponses basées sur un contexte et une question.

#### Paramètres importants :

`max_tokens=512` : Limite la longueur de la réponse.
`temperature=0.3` : Contrôle la créativité. Plus c'est bas, plus les réponses sont précises et directes.

#### Prompt "Directives du modèle"

```py
template = """<bos><start_of_turn>user
Answer the question based only on the following context and provide a detailed, accurate response...

CONTEXT: {context}
QUESTION: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
```

`Rôle` : C'est une instruction donnée au modèle pour structurer ses réponses. Ici, on demande au modèle de répondre uniquement en utilisant le contexte fourni par les documents de la base de données.

#### Streamlit pour l'interface utilisateur :

1) Fonctionnement général :
Streamlit permet de créer une interface interactive avec :

Une zone de chat pour que l'utilisateur pose ses questions.
Une zone pour afficher les réponses du chatbot.
Une sidebar pour gérer l'historique des conversations et réinitialiser le chat.

2) Affichage des messages :
Les messages de l'utilisateur apparaissent en bleu.
Les réponses de l'assistant sont en vert, avec un lien vers les sources des réponses (si disponibles).

3) Interaction en direct :

Quand l'utilisateur pose une question :

La question est ajoutée à l'historique des messages.
Le chatbot génère une réponse en cherchant dans la base de données.
La réponse (et ses sources) est affichée à l'utilisateur.