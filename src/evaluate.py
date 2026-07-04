import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def main():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X_test = pd.read_csv("data/processed/X_test.csv")['clean_message'].fillna('')
    y_test = pd.read_csv("data/processed/y_test.csv")['label']

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Évaluation terminée.")
    print(json.dumps(metrics, indent=4))
    print("\nRapport détaillé :\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
