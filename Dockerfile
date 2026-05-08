FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances
COPY requirements.txt .

# Installation des paquets Python
RUN pip install --no-cache-dir -r requirements.txt

# Téléchargement des corpus NLP pour TextBlob
RUN python -m textblob.download_corpora

# Copie du code source
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p /app/models /app/data

# Exposition du port Streamlit
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Commande de lancement
CMD ["streamlit", "run", "xgboost_trader.py", "--server.port=8501", "--server.address=0.0.0.0"]
