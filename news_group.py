from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle



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
        with open("20newsgroups.pickle", "rb") as f:
            data = pickle.load(f)


    except Exception:

        twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=['headers', 'footers', 'quotes'])
        twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42, remove=['headers', 'footers', 'quotes'])


        X_train = twenty_train.data
        y_train = twenty_train.target

        X_test = twenty_test.data
        y_test = twenty_test.target

        if lemma:
            vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), strip_accents="unicode", lowercase=True, stop_words=['english'], max_features=20000, max_df=0.25)
        elif bigram:
            vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=['english'], max_features=20000, max_df=0.25, token_pattern=r'(?u)\b[A-Za-z]\w+\b', ngram_range=(1, 2))
        else:
            vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=['english'], max_features=20000, max_df=0.25, token_pattern=r'(?u)\b[A-Za-z]\w+\b')


        # Transform train and test with vectorizer
        vectors_train = vectorizer.fit_transform(X_train)
        vectors_test = vectorizer.transform(X_test)

        data = {
            'X_train': vectors_train,
            'y_train': y_train,
            'X_test': vectors_test,
            'y_test': y_test
        }

        with open("20newsgroups.pickle", "wb") as f:
            pickle.dump(data, f)


    return data['X_train'], data['y_train'], data['X_test'], data['y_test']
