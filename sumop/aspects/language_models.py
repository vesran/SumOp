import numpy as np
import gensim


class Word2Vec:

    def __init__(self, path_to_model):
        binary = path_to_model.split('.')[-1] == 'bin'
        self.encoder = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=binary)
        self.oov_token = 'ooo'
        self.encoder[self.oov_token] = np.zeros((1, self.vector_size()))

    def __contains__(self, item):
        return item in self.encoder

    def get(self, tokens):
        if type(tokens) != list:
            tokens = [tokens]
        tokens = [word if word in self.encoder else self.oov_token for word in tokens]
        if len(tokens) == 1:
            we = self.encoder[tokens]
            #norm = np.linalg.norm(we)
            return we.reshape(1, -1) #/ max(norm, 1)
        else:
            return self.encoder[tokens]

    def vector_size(self):
        return self.encoder.vector_size

    def most_similar(self, word):
        return self.encoder.most_similar(word)
