from sumop.text_cleaning import \
    (load_symspell, cleaning_corpus, remove_punctuation, fix_spelling, custom_remove_stopwords)


def preprocess(corpus, stops):
    sym_spell = load_symspell()
    corpus = cleaning_corpus(corpus)
    corpus = [remove_punctuation(s) for s in corpus]
    corpus = fix_spelling(corpus, sym_spell)
    corpus = [custom_remove_stopwords(s, stops) for s in corpus]
    return corpus

