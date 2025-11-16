from textblob import TextBlob

def clean_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.strip()

def score_text(text):
    txt = clean_text(text)
    if txt == "":
        return 0.0
    try:
        blob = TextBlob(txt)
        return float(blob.sentiment.polarity)
    except Exception:
        return 0.0
