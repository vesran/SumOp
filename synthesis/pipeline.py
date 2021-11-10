from synthesis.params import PATH_TO_ANASENT_MODEL, PATH_TO_SAVE_SCRAPED_COMMENTS, STOPWORDS, ASPECTS, PATH_TO_W2V
from synthesis.scraper import yelp
from synthesis import anaseq
from synthesis import aspects

import tensorflow as tf


def run():
    # Scraping
    yelp.run(PATH_TO_SAVE_SCRAPED_COMMENTS)

    # Sentiment analysis
    segmenter = tf.keras.models.load_model(PATH_TO_ANASENT_MODEL)
    batch = read_text_file(PATH_TO_SAVE_SCRAPED_COMMENTS)
    # Preprocessing for sentiment analysis
    sentences = anaseq.preprocessing.preprocess(batch)
    # Sentiment analysis inference - segmenting
    fragments = anaseq.anasent.segment_corpus(sentences, segmenter, window_size=1, stops=STOPWORDS)

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
            'sentiment': fragment[1],
        })

    return fragments_data


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


if __name__ == '__main__':
    _ = run()
