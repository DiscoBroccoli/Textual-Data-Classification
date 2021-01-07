# Textual-Data-Classification
Experimenting various model performance on textual data classification with 20newsgroups [Lang,1995] & IMDB movie review dataset [Maas et al., 2011]

Completed for COMP551 - Applied Machine Learning


## Accomplishments
* Cleaning Dataset for NLP context training. Namely:
  * Bag of Words & TF-IDF.
  * Stop words, Lemmatization & N-Grams
- Hyperparameter tuning with gridsearch with 5-fold cross validation.
* Applied 7 models SciKit-Learn to obtain a holistic overview of each model's capability.
  * Logistic Regression
  * SVM
  * Naive Bayes
  * Decision Tree
  * Ada Boost
  * Random Forest 
  * MLP
  
 *After hyperparameter optimization with 5-fold cross-validation
and Grid Search, MLP and SVM achieved the highest test accuracy of 70.19% for the 20 Newsgroup
dataset and 88.15% for the IMDB Reviews dataset respectively.*
## Data Exploration Analysis

<p align="center">
<img src="https://github.com/DiscoBroccoli/Textual-Data-Classification/blob/main/20newsgroup_label.png">
</p>

The 20 Newsgroups dataset contains approximately 20,000 newsgroup documents spread across 20
different newsgroups, corresponding to 20 different topics, with some topics being closely related
to each other while others are highly unrelated. The dataset comes in the format of one file per
document with the ground truth labelled by the file directory name. The headers, footers and
quotes were omitted for each record in the dataset and the train/test partitions were retrieved
from the default SciKit-Learn split.

The large movie review dataset consists of 50,000 highly polar reviews from the IMDB website,
classified as half positive and half negative, with a 50% training and testing split. Each entry
is a single-lined text file containing a raw review. The preprocessing of the reviews consisted of
lower-casing the raw text and removing break tags so that the texts can be tokenized to create the
feature vectors.

# Feature Selection

## Bag of Words & TF-IDF

Because the algorithms cannot interpret text data directly, the Bag of Words technique was applied
to extract features for use in the models. Each text document is represented by a fixed-length
numerical vector by counting the frequency of each word. Applying Term Frequency-Inverse
Document Frequency (TF-IDF), using TfidfVectorizer from SciKit-Learn, to find the most suitable
features to describe each category. It measures the importance of each word to a document relative
to a collection of documents by comparing the word frequency in a document with the specificity
of the term (i.e. inversely proportional to its frequency in all documents). 

## Tokenization: Stop words, Lemmatization & N-Grams

For the tokenization process, the features are cleaned ant then extracted from the input data set by removing
the stop words, i.e. commonly used words in the English language. Examples of stop words include
"and", "the" and "is". Similarly, many English words appeared in the data set in inflected forms.
Therefore, two words may be written in different ways to express the same meaning. To mitigate
the issue, lemmatization technique is used, from [spaCy](https://www.researchgate.net/publication/325709583_LexNLP_Natural_language_processing_and_information_extraction_for_legal_and_regulatory_texts), which performs a dictionary
lookup to remove the inflectional endings. Lemmatizaion is a popular technique in NLP and a
team of researchers applied this technique to process [biomedical text](https://pubmed.ncbi.nlm.nih.gov/22464129/). Finally,
tokenizing n-grams is assessed, tokenized bigrams to extract a sequence of words as a feature.

```
if lemma:
  vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), strip_accents="unicode", lowercase=True,
                                         stop_words=['english'], max_features=20000, max_df=0.25)
elif bigram:
  vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=['english'],
                                         max_features=20000, max_df=0.25, token_pattern=r'(?u)\b[A-Za-z]\w+\b',
                                         ngram_range=(1, 2))
else:
  vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=['english'],
```

Code is part of [IMDB.py](https://github.com/DiscoBroccoli/Textual-Data-Classification/blob/main/imdb.py) & [news_group.py](https://github.com/DiscoBroccoli/Textual-Data-Classification/blob/main/news_group.py).

# Results

<p align="center">
<img src="https://github.com/DiscoBroccoli/Textual-Data-Classification/blob/main/20_newsgroup.png">
</p>

<p align="center">
<img src="https://github.com/DiscoBroccoli/Textual-Data-Classification/blob/main/IMDB.png">
</p>

While incorporating lemmatization and bigrams in the tokenization, the test and
validation accuracies for each dataset got worse. Thus, the results reported do not use
lemmatization and bigrams. 

The learning curve of the model plots the number of training examples against the training and
validation scores. This represents how the model learns based on its experience, or the amount of
training data (see Figures 3 and 4).Notice for most models, the training and validation
score tends to converge as the training size increases, but there are a few, notably Multilayer
Perceptron and Random Forests, where the training accuracy remains near perfect, which suggests
the tendency to overfit perfectly to the training data.

<p align="center">
<img src="https://github.com/DiscoBroccoli/Textual-Data-Classification/blob/main/table1-2.png">
</p>

<p align="center">
<img src="https://github.com/DiscoBroccoli/Textual-Data-Classification/blob/main/table3-4.png">
</p>

