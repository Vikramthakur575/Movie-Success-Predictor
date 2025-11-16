
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
from sentiment_analysis import score_text

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

# ------------------ LOAD TMDB DATA ------------------
def load_tmdb():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    # Merge on title (standard TMDB merge)
    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies.merge(credits, on="id")

    # Extract Director
    df["crew"] = df["crew"].apply(json.loads)
    df["director"] = df["crew"].apply(
        lambda x: next((c["name"] for c in x if c["job"] == "Director"), None)
    )

    # Extract top 3 cast
    df["cast"] = df["cast"].apply(json.loads)
    df["actor1"] = df["cast"].apply(lambda x: x[0]["name"] if len(x) > 0 else "")
    df["actor2"] = df["cast"].apply(lambda x: x[1]["name"] if len(x) > 1 else "")
    df["actor3"] = df["cast"].apply(lambda x: x[2]["name"] if len(x) > 2 else "")

    # Extract genres
    df["genres"] = df["genres"].apply(json.loads)
    df["genre1"] = df["genres"].apply(lambda x: x[0]["name"] if len(x) > 0 else "")

    # Extract sentiment from overview text
    df["sentiment"] = df["overview"].fillna("").apply(score_text)

    # Select features
    df = df[[
        "budget", "runtime", "popularity",
        "sentiment", "vote_average", "vote_count",
        "revenue"
    ]].fillna(0)

    return df

# ------------------ TRAIN MODEL ------------------
def train_and_save():
    df = load_tmdb()

    X = df.drop("revenue", axis=1)
    y = df["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    print("Model trained successfully!")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved as model.pkl")

if __name__ == "__main__":
    train_and_save()
