from synthesis.params import STOPWORDS

from symspellpy import SymSpell, Verbosity
from nltk.corpus import stopwords
from unidecode import unidecode
import pkg_resources
import spacy
import re

eng_stopwords = stopwords.words('english')  # python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


########################################################################
# Utils functions
########################################################################

def preprocess(batch_reviews):
    # Preprocessing
    raw_sentences = []
    # Split reviews to sentences
    for review in batch_reviews:
        sents = segment_text_to_sentences(review)
        raw_sentences += sents

    # Cleaning strings
    base_sentences = common_preprocessing(raw_sentences)

    return base_sentences


########################################################################
# Utils functions
########################################################################

def segment_text_to_sentences(text):
    chuncks = re.split(' *[.?!] *', text)
    return [sent.strip() for sent in chuncks if len(sent.strip()) > 0]


def common_preprocessing(corpus):
    sym_spell = load_symspell()
    corpus = cleaning_corpus(corpus)
    corpus = [remove_punctuation(s) for s in corpus]
    corpus = fix_spelling(corpus, sym_spell)
    return corpus


def custom_remove_stopwords(sent, stops):
    words = []
    for w in sent.split(' '):
        if w not in stops:
            words.append(w)
    return ' '.join(words)


def cleaning_corpus(corpus):
    docs = []
    for document in nlp.pipe(corpus, disable=["parser", "ner", "textcat"]):
        doc = []
        for token in document:
            doc.append(token.lemma_)
        docs.append(doc)
    return [_decontracted(' '.join(doc)) for doc in docs]


def remove_punctuation(s):
    s = s.lower()
    s = unidecode(s)
    s = re.sub('\d+', '<d>', s)
    s = re.sub("[^a-z <d>]", ' ', s)
    s = re.sub(' +', ' ', s)
    s = re.sub('ooo+', 'o', s)
    return s.strip()


def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell


def fix_spelling(corpus, sym_spell):
    clean_corpus = []
    for doc in corpus:
        words = []
        for word in doc.split(' '):
            if word == '':
                continue
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
            corrected_word = next(iter(suggestions))
            words.append(corrected_word._term)
        clean_corpus.append(' '.join(words))
    return clean_corpus


########################################################################
# Low-level functions
########################################################################

def _decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
