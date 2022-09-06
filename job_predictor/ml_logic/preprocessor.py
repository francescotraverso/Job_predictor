import string
import pandas as pd
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.models.doc2vec import TaggedDocument
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def create_corpus(df_occ_n_skills):

    # create corpus from dataframe
    X_all = pd.concat(
        [df_occ_n_skills['description_input'],
         df_occ_n_skills['skills_input']]
        ).reset_index(drop=True)

    return X_all


def read_corpus(corpus):
    # create a usable corpus from a dataframe

    # instantiate Phraser outside of the loop
    sentence_stream = [entry.split(" ") for entry in corpus]
    bigrams = Phrases(
        sentence_stream,
        min_count=5,
        threshold=5,
        connector_words=ENGLISH_CONNECTOR_WORDS)

    for i, line in enumerate(corpus):

        # remove punctuation
        for punctuation in string.punctuation:
            sentence = line.replace(punctuation, '')

        # remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(sentence)
        stopword_free_tokens = \
            [token for token in tokens if token not in stop_words]
        sentence = ' '.join(stopword_free_tokens)

        # lemmatize
        sentence = WordNetLemmatizer().lemmatize(sentence, pos='n')
        sentence = WordNetLemmatizer().lemmatize(sentence, pos='v')

        # get bigrams
        sent = sentence.split()

        # yield tagged final corpus
        yield TaggedDocument(bigrams[sent], [i])


def preprocess_input(
    sentence,
    area_keywords,
    area_kw_insert=False,
    area_kw_insert_ratio=0.2):
    # Preprocessing function for input job descriptions

    # insert data science keywords if ds_insert==True

    if area_kw_insert == True:

        sent_split = sentence.split()

        if len(sent_split) * area_kw_insert_ratio >= len(area_keywords):
            print('Warning: Chosen ratio is using up all available \
                Data Science keywords!')

        insertion_amount = int(len(sent_split) * area_kw_insert_ratio)
        insertion_counter = 0

        for insertion in range(insertion_amount):
            if len(sent_split) * area_kw_insert_ratio <= len(area_keywords):
                sent_split.append(area_keywords[insertion_counter])
                insertion_counter += 1

        sentence = ' '.join(sent_split)

    # remove punctuation
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    # set lowercase
    sentence = sentence.lower()

    # remove numbers
    sentence = ''.join(char for char in sentence if not char.isdigit())

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(sentence)
    stopword_free_tokens = \
        [token for token in tokens if token not in stop_words]
    sentence = ' '.join(stopword_free_tokens)

    # lemmatize
    sentence = WordNetLemmatizer().lemmatize(sentence, pos='n')
    sentence = WordNetLemmatizer().lemmatize(sentence, pos='v')

    return sentence


def truncate_description(job_description, no_words=100):
    return ' '.join(job_description.split()[:no_words:1])
