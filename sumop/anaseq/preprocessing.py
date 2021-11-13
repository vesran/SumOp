from sumop.text_cleaning import \
    (segment_text_to_sentences, load_symspell, cleaning_corpus, remove_punctuation, fix_spelling)


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


def common_preprocessing(corpus):
    sym_spell = load_symspell()
    corpus = cleaning_corpus(corpus)
    corpus = [remove_punctuation(s) for s in corpus]
    corpus = fix_spelling(corpus, sym_spell)
    return corpus

