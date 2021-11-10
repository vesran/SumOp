from synthesis.params import PATH_TO_ANASENT_MODEL, STOPWORDS, ASPECTS, PATH_TO_W2V
from synthesis import anaseq
from synthesis import aspects

import tensorflow as tf


class Pipeline:

    def __init__(self, window_size=1, seed_size=100, aspects=None):
        """
        :param window_size: int specifying the window size for interpolation of polarity
        :param seed_size: int specifying the number of words to consider as seed to detect an aspect
        :param aspects: dict containing aspect string -> proba threshold used to decide either the aspect is concerned
        or not
        """
        self.embedder = aspects.language_models.Word2Vec(PATH_TO_W2V)
        self.segmenter = tf.keras.models.load_model(PATH_TO_ANASENT_MODEL)
        self.window_size = window_size
        self.seed_size = seed_size
        self.aspect = aspects if aspects is not None else ASPECTS

    def __call__(self, batch, *args, **kwargs):
        """
        :param batch: Corpus / Batch / Lists of raw reviews
        :return: dict containing text fragment -> metadata (sentiment, aspect proba)
        """
        # Preprocessing for sentiment analysis
        sentences = anaseq.preprocessing.preprocess(batch)
        # Sentiment analysis inference - segmenting
        fragments = anaseq.anasent.segment_corpus(sentences, self.segmenter, window_size=1, stops=STOPWORDS)

        # Aspects detections
        embedder = aspects.language_models.Word2Vec(PATH_TO_W2V)
        # Preprocessing for aspect detection
        fragments_text = [fragment[0] for fragment in fragments]
        clean_fragments_text = aspects.preprocessing.preprocess(fragments_text, STOPWORDS)
        # Aspect detection
        fragments_data = aspects.recat.run(list(ASPECTS.keys()), embedder, clean_fragments_text)

        # Reformat data
        for data, fragment in zip(fragments_data, fragments):
            for aspect in ASPECTS:
                data.update({aspect: 1 if data[aspect] >= ASPECTS[aspect] else 0})
            data.update({
                'text': fragment[0],
                'sentiment': self._decode_sentiment_label(fragment[1]),
            })
        return fragments_data

    def _decode_sentiment_label(self, label):
        if label == 1:
            return 'POSITIVE'
        elif label == 0:
            return 'NEGATIVE'
        else:
            raise ValueError(f'Label {label} for sentiment is unknown.')

