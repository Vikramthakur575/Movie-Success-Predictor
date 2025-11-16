import requests

try:
    r = requests.get("https://api.themoviedb.org/3/configuration", timeout=5)
    print("Status:", r.status_code)
    print("Success, TMDB reachable.")
except Exception as e:
    print("Failed:", e)
