import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
import string



nltk.download('stopwords')
parfums_carac=pd.read_csv(r'C:\Users\moham\OneDrive\Desktop\recommendation_model\DATA\parfums_300\parfums_300.csv')
avis_user=pd.read_csv(r'C:\Users\moham\OneDrive\Desktop\recommendation_model\DATA\avis_300\avis_300.csv')


merged_df = pd.merge(avis_user, parfums_carac, left_on='id_parfum', right_index=True, how='inner')
print(merged_df.head())

# data_superficial_cleaning
merged_df["commentaire"] = merged_df["commentaire"].astype(str).str.replace("/", " ")

# Remove punctuation
merged_df["commentaire"] = merged_df["commentaire"].str.translate(str.maketrans('', '', string.punctuation))

# Remove digits
merged_df["commentaire"] = merged_df["commentaire"].str.replace(r'\d+', '', regex=True)

# Remove running spaces
merged_df["commentaire"]= merged_df["commentaire"].str.replace(r'\s+', ' ', regex=True).str.strip()

# Make the text lowercase
merged_df["commentaire"] = merged_df["commentaire"].str.lower()
print(merged_df["commentaire"])


#stop words eliminate

stop_words = set(stopwords.words('french')) # Assuming French text

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


merged_df["commentaire"] = merged_df["commentaire"].apply(remove_stopwords)



#lemmetization


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Télécharger les ressources nécessaires
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Initialiser le lemmatiseur
lemmatizer = WordNetLemmatizer()

# Charger les stopwords français
stop_words = set(stopwords.words('french'))

# Fonction pour lemmatiser une chaîne de caractères
def lemmatize_text(text):
    tokens = word_tokenize(text, language='french')  # Tokenization
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]  # Lemmatisation
    return " ".join(lemmas)  # Rejoindre les lemmes en une seule chaîne

# Appliquer la fonction à chaque ligne de la colonne
merged_df["commentaires_lemmatisees"] = merged_df["commentaire"].apply(lemmatize_text)

# Afficher les résultats
print(merged_df[["commentaire", "commentaires_lemmatisees"]].head())