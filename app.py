import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Chargement du modèle et du vectorizer
with open("models/model_final.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Préparation des stop words (négations conservées, comme dans le prétraitement d'origine)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negations = {"no", "not", "nor", "n't", "cannot", "can't", "won't", "don't",
             "isn't", "wasn't", "aren't", "weren't", "didn't", "doesn't",
             "hasn't", "haven't", "hadn't", "wouldn't", "shouldn't", "couldn't"}
stop_words = stop_words - negations

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'&quot;|&amp;|&lt;|&gt;', ' ', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Interface Streamlit
st.title("🔍 Détection de tweets suspects")
st.write("Cette application prédit si un tweet est **suspect** ou **normal**, à partir d'un modèle Random Forest entraîné sur des données textuelles.")

tweet_input = st.text_area("Saisissez un tweet :", height=100)

if st.button("Prédire"):
    if tweet_input.strip() == "":
        st.warning("Merci de saisir un tweet avant de prédire.")
    else:
        cleaned = clean_text(tweet_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        label = "🚨 Suspect" if prediction == 0 else "✅ Normal"
        proba_suspect = probabilities[0]
        proba_normal = probabilities[1]

        st.subheader(f"Résultat : {label}")
        st.write(f"**Probabilité suspect :** {proba_suspect:.2%}")
        st.write(f"**Probabilité normal :** {proba_normal:.2%}")

        st.progress(float(proba_suspect))
