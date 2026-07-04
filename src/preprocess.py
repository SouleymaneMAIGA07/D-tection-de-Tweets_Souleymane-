import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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

def main():
    df = pd.read_csv("tweets_dataset.csv")
    df['clean_message'] = df['message'].apply(clean_text)
    df = df[df['clean_message'].str.strip() != ''].reset_index(drop=True)
    df.to_csv("data/processed/clean_dataset.csv", index=False)
    print(f"Prétraitement terminé : {len(df)} lignes sauvegardées.")

if __name__ == "__main__":
    main()
