import imdb_data
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle
import numpy as np



class LemmaTokenizer(object):
    def __init__(self):
        spacy.load('en')
        self.lemmatizer_ = spacy.lang.en.English()
    def __call__(self, docs):
        tokens = self.lemmatizer_(docs)
        return [t.lemma_ for t in tokens if t.is_alpha]


# ________________________________________________
# Vectorized train and test splits
def fetch_data(reset=False, lemma=False, bigram=False):
    try:
        if reset:
            raise Exception
        with open("imdb.pickle", "rb") as f:
            data = pickle.load(f)


    except Exception:

        imdb_train = imdb_data.load_train()
        imdb_test = imdb_data.load_test()
        np.random.shuffle(imdb_train)
        np.random.shuffle(imdb_test)

        X_train, y_train = zip(*imdb_train)

        X_test, y_test = zip(*imdb_test)

        if lemma:
            vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), strip_accents="unicode", lowercase=True,
                                         stop_words=['english'], max_features=20000, max_df=0.25)
        elif bigram:
            vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=['english'],
                                         max_features=20000, max_df=0.25, token_pattern=r'(?u)\b[A-Za-z]\w+\b',
                                         ngram_range=(1, 2))
        else:
            vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=['english'],
                                         max_features=20000, max_df=0.25, token_pattern=r'(?u)\b[A-Za-z]\w+\b')

        # Transform train and test with vectorizer
        vectors_train = vectorizer.fit_transform(X_train)
        vectors_test = vectorizer.transform(X_test)

        data = {
            'X_train': vectors_train,
            'y_train': y_train,
            'X_test': vectors_test,
            'y_test': y_test
        }

        with open("imdb.pickle", "wb") as f:
            pickle.dump(data, f)


    return data['X_train'], data['y_train'], data['X_test'], data['y_test']
