from synthesis.params import PATH_TO_ANASENT_MODEL, PATH_TO_SAVE_SCRAPED_COMMENTS, STOPWORDS
from synthesis.scraper import yelp
from synthesis import anaseq

import tensorflow as tf


def run():
    # Scraping
    yelp.run(PATH_TO_SAVE_SCRAPED_COMMENTS)

    # Sentiment analysis
    segmenter = tf.keras.models.load_model(PATH_TO_ANASENT_MODEL)
    batch = read_text_file(PATH_TO_SAVE_SCRAPED_COMMENTS)

    # Preprocessing
    sentences = anaseq.preprocessing.preprocess(batch)

    # Inference
    fragments = anaseq.anasent.segment_corpus(sentences, segmenter, window_size=1, stops=STOPWORDS)


########################################################################
# Utils functions
########################################################################

def read_text_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    reviews = []
    for line in lines:
        review = line.strip()
        reviews.append(review)

    return reviews


