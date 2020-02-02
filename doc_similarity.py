#%%
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import nltk
import re

#%%
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)

#%%
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
    text = re.sub(r'#+', ' ', text )
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
    text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)
    #text = re.sub(r"\'s", " ", text)     text = re.sub(r"\'ve", " have ", text)
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

data = ['Tell me about yourself','I recently graduated with a degree in computer science and am really excited about the prospect of this position. I have interned with a few mid-size companies and was able to work closely with their mobile developers. I also have a lot of experience with online code repositories, and I love to contribute as much as I can.', 'How would people describe you?','I would say that people would describe me as a good communicator because I’m always able to articulate people’s needs, whether it be clients or my teammates. I’m able to quickly pick up on subtle hints and nonverbal clues, and I believe this has helped contribute to my success.', 'Which is your most significant accomplishment?', 'When I first started at my internship, the onboarding process wasn’t very thorough, and the initial training for developers left a lot to be desired. After sharing my concerns with my trainer, I was able to help develop better resources for new employees as well as restructure the program entirely. I feel like this showed both my initiative and my problem-solving abilities.', 'What is your greatest strength?', 'I’m highly motivated and am extremely passionate about developing. I am known for completing projects ahead of deadlines, and I feel this is especially important when things are constantly evolving and changing. I take initiative and am always coming up with various ways to solve a problem without needing to wait for direction. I also am always up-to-date with the newest trends and am willing to try the latest thing.','What is your greatest weakness?', 'I would say my biggest weakness is that I’m a recent graduate and don’t have a lot of work experience, but I’m a fast-learner and am highly adaptable. I’m up-to-speed with the latest programming trends and have a fresh perspective. I know I have enthusiasm for the work, and I will bring my strong work ethic and commitment every day.', 'What are your preferred programming languages and why?', 'My preferred programming language is Python because it’s easy for a beginner to pick up and it has a lot of excellent libraries. The support libraries greatly reduce time spent on coding, and I find it helps me be creative yet efficient.', 'What makes me unique is my experience of having spent four years in retail. Because I’ve had first-hand experience fielding shoppers’ questions, feedback and complaints, I know what customers want. I know what it takes to create a positive consumer experience because I’ve had that direct interaction, working directly with consumers in person.', 'Making a meaningful difference in the lives of my patients and their families motivate me to strive for excellence in everything I do. I look forward to seeing their reaction when we get a positive outcome that will change their lives forever. Like the family of a young boy we treated last year. At eight years old, he had experienced rapid weight gain and signs of depression. His parents described him as a usually joyful child, but now he seemed disengaged and uninterested in his typical schedule. In the end, we determined that it was hypothyroidism which is, of course, controllable with medication. The boy is adjusting well to the treatment and has returned to his joyful self. That’s why I became a nurse and why I’m pursuing a position in pediatrics.']

data_processed = list(map(process_text, data))

#%%
def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return sess.run(embed(texts))
#%%
BASE_VECTORS = get_features(data)

#%%
BASE_VECTORS.shape

#%%
def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

#%%
def test_similiarity(text1, text2):
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]
    print(vec1.shape)
    return cosine_similarity(vec1, vec2)

