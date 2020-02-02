from sentiment_analyzer import analyze_sentiment
from targetting_analyzer import analyze_targetting


def analyze(actual_text, optimal_text):
    return analyze_sentiment(actual_text), analyze_targetting(actual_text, optimal_text)
