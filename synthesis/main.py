from synthesis.params import PATH_TO_SAVE_SCRAPED_COMMENTS
from synthesis.pipeline import Pipeline
from synthesis.scraper import yelp

import pandas as pd


def main():
    # Scraping
    yelp.run(PATH_TO_SAVE_SCRAPED_COMMENTS)
    text_batch = read_text_file(PATH_TO_SAVE_SCRAPED_COMMENTS)

    # Inference
    model = Pipeline()
    data = model(text_batch)

    df = pd.DataFrame(data)

    return df


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
    main()
