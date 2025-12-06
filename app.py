import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# ----------------------------------
# Load dataset
# ----------------------------------
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")
movies = movies.merge(credits, on="title")

model = pickle.load(open("model.pkl", "rb"))
import os

# Load API key from Streamlit Secrets (Streamlit Cloud)
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("❌ TMDB_API_KEY missing! Add it in Streamlit → Settings → Secrets.")



nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ---------------------------------------------------
# TMDB API FETCH FUNCTION
# ---------------------------------------------------
def fetch_tmdb_data(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}

    try:
        res = requests.get(url, params=params, timeout=8)
        res.raise_for_status()
        data = res.json()

        if not data["results"]:
            return 0, 0, "No overview available", None

        movie = data["results"][0]

        rating = movie.get("vote_average", 0)
        votes = movie.get("vote_count", 0)
        overview = movie.get("overview", "No overview available")
        poster_path = movie.get("poster_path", None)

        poster_url = (
            f"https://image.tmdb.org/t/p/w500{poster_path}"
            if poster_path else None
        )

        return rating, votes, overview, poster_url

    except:
        return 0, 0, "API Error", None

# ---------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------
def recommend(movie_title):
    try:
        idx = movies[movies["title"] == movie_title].index[0]
    except:
        return []

    movies["similarity"] = movies["overview"].fillna("").apply(
        lambda x: abs(len(x) - len(movies.loc[idx, "overview"]))
    )

    top6 = movies.sort_values("similarity").head(6)["title"].tolist()

    return [m for m in top6 if m != movie_title][:5]

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🎬 Movie Success Predictor + EDA Dashboard")

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 EDA Dashboard"])

# ---------------------------------------------------
# TAB 1 — PREDICTION
# ---------------------------------------------------
with tab1:

    movie_list = movies["title"].sort_values().unique()
    selected_movie = st.selectbox("Search a Movie:", movie_list)

    if selected_movie:
        ov = movies[movies["title"] == selected_movie]["overview"].values[0]
        sentiment = sia.polarity_scores(str(ov))["compound"]

        rating, votes, overview, poster = fetch_tmdb_data(selected_movie)

        row = movies[movies["title"] == selected_movie].iloc[0]

        input_df = pd.DataFrame({
            "budget": [row["budget"]],
            "runtime": [row["runtime"]],
            "popularity": [row["popularity"]],
            "sentiment": [sentiment],
            "vote_average": [rating],
            "vote_count": [votes]
        })

        predicted_rev = model.predict(input_df)[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.warning("No Poster Available")

        with col2:
            st.subheader(selected_movie)
            st.write(f"💰 **Predicted Revenue:** ${predicted_rev:,.2f}")
            st.write(f"⭐ **Rating:** {rating}")
            st.write(f"🗳 **Votes:** {votes}")
            st.write(f"😊 **Sentiment:** {sentiment:.3f}")
            st.write(f"📝 **Overview:** {overview}")

        st.subheader("🎯 Recommended Movies")
        for r in recommend(selected_movie):
            st.write("👉", r)

# ---------------------------------------------------
# TAB 2 — EDA
# ---------------------------------------------------
with tab2:
    st.header("📊 Exploratory Data Analysis")

    # 1️⃣ Budget Distribution
    st.subheader("💰 Budget Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(movies["budget"], bins=40)
    ax1.set_xlabel("Budget")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # 2️⃣ Popularity vs Revenue
    st.subheader("🔥 Popularity vs Revenue")
    fig2, ax2 = plt.subplots()
    ax2.scatter(movies["popularity"], movies["revenue"], alpha=0.3)
    ax2.set_xlabel("Popularity")
    ax2.set_ylabel("Revenue")
    st.pyplot(fig2)

    # 3️⃣ Runtime Distribution
    st.subheader("⏱ Runtime Distribution")
    fig3, ax3 = plt.subplots()
    ax3.hist(movies["runtime"].dropna(), bins=40, color="orange")
    ax3.set_xlabel("Runtime")
    ax3.set_ylabel("Movies Count")
    st.pyplot(fig3)

    # 4️⃣ Top 20 Most Popular Movies
    st.subheader("⭐ Top 20 Most Popular Movies")
    top_popular = movies.nlargest(20, "popularity")[["title", "popularity"]]
    st.dataframe(top_popular)

