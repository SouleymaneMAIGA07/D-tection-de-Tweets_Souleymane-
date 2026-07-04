import pandas as pd
import yaml
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open("params.yaml") as f:
    params = yaml.safe_load(f)['train']

def main():
    df = pd.read_csv("data/processed/clean_dataset.csv")
    df['clean_message'] = df['clean_message'].fillna('')

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_message'], df['label'],
        test_size=params['test_size'],
        random_state=params['random_state'],
        stratify=df['label']
    )

    vectorizer = TfidfVectorizer(max_features=params['max_features'])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=params['random_state'])
    model.fit(X_train_vec, y_train)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Sauvegarde du jeu de test pour l'évaluation
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Entraînement terminé. Modèle sauvegardé dans models/model.pkl")

if __name__ == "__main__":
    main()
