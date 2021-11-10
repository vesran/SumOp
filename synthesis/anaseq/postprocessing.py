# ndoc = (the, 0), (world, 1), ...

########################################################################
# Main functions
########################################################################

def segment_corpus(corpus, segmenter, window_size, stops):
    sequences = transform_corpus_to_ndocs_with_predictions(corpus, segmenter, window_size, stops)
    sequences = [split_prediction(ndoc) for ndoc in sequences]
    units = [transform_ndoc_unit_to_text(ndoc) for ndoc in sequences]
    sentiments = [get_units_sentiment(ndoc) for ndoc in sequences]
    return units, sentiments


def transform_corpus_to_ndocs_with_predictions(corpus, segmenter, window_size, stops):
    ndocs = number_documents(corpus)
    ndocs_2 = remove_stopwords_batch(ndocs, stops)
    ndocs_3 = predict_batch(ndocs_2, segmenter)
    ndocs_4 = merge_ndocs_with_prediction(ndocs, ndocs_3)
    ndocs_5 = complete_prediction(ndocs_4, window_size)
    return ndocs_5


def split_prediction(ndoc):
    if all(len(item) == 2 for item in ndoc):  # Don't exists at least one prediction
        print(f"No way to complete : {ndoc}")
        return [ndoc]

    units = [[]]
    for w, i, p in ndoc:
        if len(units[-1]) == 0 or ((units[-1][-1][2] - 0.5) * (p - 0.5) > 0):  # Pas de changement de signe
            units[-1].append((w, i, p))
        else:
            units.append([(w, i, p)])
    return units


def transform_ndoc_unit_to_text(ndoc_units):
    texts = []
    for ndoc in ndoc_units:
        if all(len(item) == 2 for item in ndoc):
            text = ' '.join([w for w, _ in ndoc])
        else:
            text = ' '.join([w for w, _, _ in ndoc])
        texts.append(text)
    return texts


def get_units_sentiment(ndoc_units):
    return [int(np.around(ndoc[0][-1])) for ndoc in ndoc_units]


########################################################################
# Low-level functions
########################################################################

def number_documents(documents):
    return [[(w, i) for i, w in enumerate(document.split(' '))] for document in documents]


def remove_stopwords_batch(ndocs, stops):
    return [list(filter(lambda x: not x[0] in stops, ndoc)) for ndoc in ndocs]


def predict_batch(ndocs, segmenter):
    docs = np.array([' '.join([w for w, _ in ndoc]) for ndoc in ndocs])
    preds = segmenter.predict(docs).reshape((len(docs), -1))
    ndocs_preds = [[(w, i, p) for (w, i), p in zip(ndoc, pred)] for ndoc, pred in zip(ndocs, preds)]
    return ndocs_preds


def merge_ndocs_with_prediction(ndocs, ndocs_preds):
    for i in range(len(ndocs)):
        for w, j, p in ndocs_preds[i]:
            ndocs[i][j] = (w, j, p)
    return ndocs


def complete_prediction_single_ndoc(ndoc, window_size):
    if all(len(item) == 2 for item in ndoc):  # Exists at least one prediction
        print(f"No way to complete : {ndoc}")
        return ndoc
    while any(len(item) == 2 for item in ndoc):
        for i in range(len(ndoc)):
            if len(ndoc[i]) == 3:
                continue
            start, end = max(i-window_size, 0), min(i+window_size+1, len(ndoc))
            values = [ndoc[i][2] for i in range(start, end) if len(ndoc[i]) == 3]
            if len(values) > 0:
                ndoc[i] = (ndoc[i][0], ndoc[i][1], np.mean(values))
    return ndoc