
import pytest
import app
import pandas as pd
from unittest.mock import patch, MagicMock


# -------------------------
# Test TMDB Fetch Function
# -------------------------
def test_fetch_tmdb_success():
    mock_response = {
        "results": [
            {"vote_average": 7.5, "vote_count": 1200}
        ]
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None

        rating, votes = app.fetch_tmdb("Inception")

    assert rating == 7.5
    assert votes == 1200


def test_fetch_tmdb_no_results():
    mock_response = {"results": []}

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None

        rating, votes = app.fetch_tmdb("UnknownMovieXYZ")

    assert rating == 0
    assert votes == 0


# -------------------------
# Test Model Prediction
# -------------------------
def test_prediction_pipeline():
    mock_model = MagicMock()
    mock_model.predict.return_value = [123456789]

    with patch("app.model", mock_model):

        row = pd.DataFrame({
            "budget": [10000000],
            "runtime": [120],
            "popularity": [500],
            "sentiment": [0.8],
            "vote_average": [7.2],
            "vote_count": [1000]
        })

        result = mock_model.predict(row)[0]

    assert result == 123456789
