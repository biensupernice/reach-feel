from sentiment_analyzer import analyze_sentiment
from targetting_analyzer import analyze_targetting


def analyze(text):
    sentiment = analyze_sentiment(text)
    targetting = analyze_targetting(text)
    return sentiment
