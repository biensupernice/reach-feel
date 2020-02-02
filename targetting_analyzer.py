from doc_similarity import get_similarity


def analyze_targetting(actual_text, optimal_text):
    return get_similarity(actual_text, optimal_text)
