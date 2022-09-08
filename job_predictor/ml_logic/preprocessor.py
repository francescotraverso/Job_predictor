import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from job_predictor.ml_logic.nltkmodules import stopwords


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

        for insertion in range(insertion_amount):
            if len(sent_split) * area_kw_insert_ratio <= len(area_keywords):
                sent_split.append(area_keywords[insertion])

        sentence = ' '.join(sent_split)

    # remove punctuation
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    # set lowercase
    sentence = sentence.lower()

    # remove numbers
    sentence = ''.join(char for char in sentence if not char.isdigit())

    # remove stopwords
    stop_words = stopwords
    tokens = word_tokenize(sentence)
    stopword_free_tokens = \
        [token for token in tokens if token not in stop_words]
    sentence = ' '.join(stopword_free_tokens)

    # lemmatize --- disabled!
    # sentence = WordNetLemmatizer().lemmatize(sentence, pos='n')
    # sentence = WordNetLemmatizer().lemmatize(sentence, pos='v')

    return sentence


def truncate_description(job_description, no_words=100):
    return ' '.join(job_description.split()[:no_words:1])
