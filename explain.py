
# explain.py
import joblib
import pandas as pd
import shap
from sentiment_analysis import score_text

MODEL_PATH = "models/movie_model.joblib"
FEATURES_PATH = "models/features.txt"

# Load model & features
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

# Create explainer once (uses model.predict)
explainer = shap.Explainer(model.predict, feature_names=FEATURES)

def explain_prediction(budget, popularity, runtime, overview):
    """
    Returns shap_values and the input dataframe (X) used for explanation.
    shap_values is a shap.Explanation object (can be plotted).
    """
    sentiment = score_text(overview)
    X = pd.DataFrame([{
        "budget": float(budget),
        "popularity": float(popularity),
        "runtime": float(runtime),
        "overview_sentiment": float(sentiment)
    }])[FEATURES]
    shap_values = explainer(X)
    return shap_values, X
