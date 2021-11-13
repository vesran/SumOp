from sumop.params import ASPECTS_PARAMS

from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from collections import Counter
import numpy as np
import spacy


########################################################################
# Main functions
########################################################################

def run(aspects, embedder, corpus):
    aspects2predictions = {}
    for aspect in aspects:
        seed, _ = extend_seed_words([aspect], embedder, limit=ASPECTS_PARAMS['limit'])
        predictions = predict_single_aspect(corpus, seed, aspects, embedder)
        aspects2predictions[aspect] = predictions

    # Reformat in a JSON format
    data = []
    for i, text in enumerate(corpus):
        review_info = {'text': text}
        for aspect in aspects:
            review_info.update({aspect: aspects2predictions[aspect][i]})
        data.append(review_info)
    return data



########################################################################
# Utils
########################################################################

def extract_frequent_nouns(corpus, top, embedder):
    nlp = spacy.load("en_core_web_sm")
    not_nouns = set()
    aspect_words = Counter()
    for document in nlp.pipe(corpus, disable=["parser", "ner", "textcat"]):
        for token in document:
            if token.pos_ == 'NOUN' and token.text in embedder and token.text not in not_nouns:
                aspect_words[token.text] += 1
            elif token.pos_ != 'NOUN':
                not_nouns.add(token.text)
    return [w for w, _ in aspect_words.most_common(top)]


def extend_seed_words(seed_words, w2v_model, limit=100):
    limit_single = limit // len(seed_words)
    assert limit_single >= 1
    final_seed = []
    index = []
    for seed_word in seed_words:
        seed_word_extension = []
        words_to_extend = [seed_word]
        while len(seed_word_extension) < limit_single and len(words_to_extend) > 0:
            word = words_to_extend.pop(0)
            if word not in w2v_model:
                continue
            simil_words = [t[0] for t in w2v_model.most_similar(word)]
            words_to_extend.extend(simil_words)
            seed_word_extension.append(word) if word not in seed_word_extension else 0
        final_seed.extend(seed_word_extension)
        index = index + [seed_words.index(seed_word)] * len(seed_word_extension)
    assert len(index) == len(final_seed)
    return final_seed, np.array(index)


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    s = e_x.sum(axis=axis, keepdims=True)
    return e_x / s


def compute_attention(S, A, gamma=None):
    # mat = (cosine_similarity(S, A) + 1) / 2  # Cosine !!!! (instead of RBF)
    mat = rbf_kernel(S, A, gamma)
    smat = mat.sum(axis=1)
    s = mat.sum()
    if s == 0:
        # If s happens to be 0, back off to uniform
        return softmax(np.ones((1, len(S))) / len(S))
    else:
        atts = softmax(smat.reshape(1, -1) / s)
        return atts


def compute_doc_cat(S, A, gamma=None):
    atts = compute_attention(S, A, gamma=gamma)
    d = (atts.T * S).sum(axis=0)
    return d


def predict_cat(corpus, labels, aspects, embedder, gamma=None):
    # Aspect matrix
    A = np.zeros((len(aspects), embedder.vector_size()))
    for i, a in enumerate(aspects):
        A[i] = embedder.get(a)
    print(A.shape)

    # Labels embeddings
    L = np.zeros((len(labels), 200))
    for j in range(len(labels)):
        L[j] = embedder.get(labels[j])

    D = np.zeros((len(corpus), embedder.vector_size()))
    for i, sent in enumerate(corpus):
        # Seq embeddings
        instance = sent.split(' ')
        S = np.zeros((len(instance), 200))
        for j, w in enumerate(instance):
            S[j] = embedder.get(w)[0]

        # Document embeddings
        D[i][:] = compute_doc_cat(S, A, gamma=gamma)

    return cosine_similarity(D, L)


def predict_catex(corpus, aspects_labels, aspects, ids, embedder, gamma=None):
    # Aspect matrix
    A = np.zeros((len(aspects), embedder.vector_size()))
    for i, a in enumerate(aspects):
        A[i] = embedder.get(a)

    # Labels embeddings - not limited to 3
    L = np.zeros((len(aspects_labels), 200))

    for j in range(len(aspects_labels)):
        L[j] = embedder.get(aspects_labels[j])

    D = np.zeros((len(corpus), embedder.vector_size()))
    for i, sent in enumerate(corpus):
        # Seq embeddings
        instance = sent.split(' ')
        S = np.zeros((len(instance), 200))
        for j, w in enumerate(instance):
            S[j] = embedder.get(w)[0]

        # Document embeddings
        D[i][:] = compute_doc_cat(S, A, gamma=gamma)

    sims = cosine_similarity(D, L)
    maxes = sims.argmax(axis=1)
    preds = ids[maxes]

    return preds


def predict_single_aspect(corpus, aspect_labels, aspects, embedder, gamma=None):
    # Aspect matrix
    A = np.zeros((len(aspects), embedder.vector_size()))
    for i, a in enumerate(aspects):
        A[i] = embedder.get(a)

    # Labels embeddings - not limited to 3
    L = np.zeros((len(aspect_labels), 200))

    for j in range(len(aspect_labels)):
        L[j] = embedder.get(aspect_labels[j])

    D = np.zeros((len(corpus), embedder.vector_size()))
    for i, sent in enumerate(corpus):
        # Seq embeddings
        instance = sent.split(' ')
        S = np.zeros((len(instance), 200))
        for j, w in enumerate(instance):
            S[j] = embedder.get(w)[0]

        # Document embeddings
        D[i][:] = compute_doc_cat(S, A, gamma=gamma)

    sims = cosine_similarity(D, L)
    return np.max(sims, axis=1)
