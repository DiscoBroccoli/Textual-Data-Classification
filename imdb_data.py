import re
import os
import os.path as op
from joblib import dump, load
from pathlib import Path

DATA_DIR = op.dirname(__file__)

TEST_DIR = op.join(DATA_DIR, 'data', 'test')
TRAIN_DIR = op.join(DATA_DIR, 'data', 'train')
MODEL_DIR = op.join(DATA_DIR, 'model')

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

def load_data(directory: str) -> list:
    data = []
    categories = {'neg': 0, 'pos': 1}
    for category in os.listdir(directory):
        if category not in categories:
            continue
        category_bin = categories[category]
        for filename in os.listdir(op.join(directory, category)):
            with open(op.join(directory, category, filename), 'rb') as f:
                review = clean_review(f.read().decode('utf-8').lower())
                data.append([review, category_bin])
    return data # [[comment as string, label], ...]

def load_test() -> list:
    return load_data(TEST_DIR)

def load_train() -> list:
    return load_data(TRAIN_DIR)

def clean_review(review):
    replacements = {
        '<br />': ' '
        #, "'m": ' am', "'s": ' is', "'t": ' not',
        #"'d": ' would', "'re": ' are', "'ll": ' will', "'ve": ' have'
    }

    regexp = re.compile('|'.join(map(re.escape, replacements)))
    return regexp.sub(lambda match: replacements[match.group(0)], review)

