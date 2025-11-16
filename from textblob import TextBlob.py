from textblob import TextBlob

text = "I love this movie!"
print(TextBlob(text).sentiment.polarity)  # Should print a value > 0
