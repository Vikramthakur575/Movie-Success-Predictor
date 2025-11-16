import requests, os
from dotenv import load_dotenv
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

res = requests.get(
    "https://api.themoviedb.org/3/search/movie",
    params={"api_key": TMDB_API_KEY, "query": "Inception"}
)
print(res.status_code)
print(res.json())
