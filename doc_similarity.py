import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import nltk
import re

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)


def remove_stopwords(stop_words, tokens):
    res = []
    for token in tokens:
        if not token in stop_words:
            res.append(token)
    return res


def process_text(text):
    text = text.encode('ascii', errors='ignore').decode()
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'#+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
    text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


def lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemma_list = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, 'v')
        if lemma == token:
            lemma = lemmatizer.lemmatize(token)
        lemma_list.append(lemma)
    return lemma_list


def process_all(text, stop_words):
    text = process_text(text)
    return ' '.join(remove_stopwords(stop_words, text.split()))


def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    g = tf.Graph()
    with g.as_default():
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        embed = hub.Module(module_url)
        with tf.Session(graph=g) as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.tables_initializer()])
            return sess.run(embed(texts))


def get_similarity(text_a, text_b):
    pr_a = process_text(text_a)
    pr_b = process_text(text_b)
    return test_similiarity(pr_a, pr_b)


def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)


def test_similiarity(text1, text2):
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]
    return cosine_similarity(vec1, vec2)
