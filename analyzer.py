from sentiment_analyzer import analyze_sentiment
from targetting_analyzer import analyze_targetting


def analyze(actual_text, optimal_text):
    return analyze_sentiment(actual_text), analyze_targetting(actual_text, optimal_text)

def score_states(score):
    if score >= .8:
        return 'Excellent! Your response was focused and addressed the question well.'
    elif score >= .65 and score < .8:
        return 'Good job. Your response was slightly off topic and could be improved. Malke sure not to stray away from what the question is asking'
    else:
        return 'Hm. It seems your response could use some work. Make sure to read the question carefully and address the main idea. You got this!'