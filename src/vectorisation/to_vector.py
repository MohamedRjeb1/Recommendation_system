
from transformers import BertTokenizer, BertModel
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Assurez-vous que le modèle et le tokenizer sont téléchargés
nltk.download("vader_lexicon")

# Initialisation du tokenizer et du modèle BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Fonction pour obtenir le vecteur de phrase à partir de BERT
def vectoriser_avis_bert(avis):
    inputs = tokenizer(avis, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Obtenir le vecteur de la dernière couche cachée et faire la moyenne
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Chargement des données
from src.data_cleaning.data_clean import merged_df

# Extraction des avis
avis_clients = merged_df["commentaires_lemmatisees"].tolist()

# Appliquer la vectorisation sur tous les avis avec BERT
merged_df["avis_vectorisés"] = merged_df["commentaires_lemmatisees"].apply(lambda x: vectoriser_avis_bert(" ".join(x)))

# Analyse de sentiment avec VADER
sia = SentimentIntensityAnalyzer()

# Appliquer l'analyse de sentiment sur les avis originaux
merged_df["sentiment_score"] = merged_df["commentaire"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])

# Ajouter une catégorie (Positif / Négatif / Neutre)
merged_df["sentiment"] = merged_df["sentiment_score"].apply(lambda x: "Positif" if x > 0.05 else ("Négatif" if x < -0.05 else "Neutre"))

# Afficher les résultats
print(merged_df[["commentaire", "sentiment_score", "sentiment"]])
