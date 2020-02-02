from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


def analyze_sentiment(text):
    return extract_average_sentiment(
        language
        .LanguageServiceClient()
        .analyze_sentiment(
            document=types.Document(
                content=text,
                type=enums.Document.Type.PLAIN_TEXT))
        .sentences)


def extract_average_sentiment(sentences):
    total_sentiment_score = 0
    for sentence in sentences:
        total_sentiment_score += sentence.sentiment.score
    return total_sentiment_score / len(sentences)
