from synthesis.params import FORBIDDEN_TEXT, URL_FORMAT, PATH_TO_SAVE_SCRAPED_COMMENTS

from bs4 import BeautifulSoup
import pandas as pd
import urllib
import re


###########################################################################
# MAIN
###########################################################################
def main():
    reviews, ratings = scrap_yelp()

    # Saving to csv
    df = pd.DataFrame({'review': reviews, 'rating': ratings})
    df.to_csv(PATH_TO_SAVE_SCRAPED_COMMENTS, index=False, sep=';')


###########################################################################
# PROCESSING FUNCTIONS
###########################################################################

def gen_url_yelp(url_format):
    for i in range(10, 201, 10):
        number = str(i)
        url = re.sub('#', number, url_format)
        yield url


def get_soup(url):
    # query the website and return the html to the variable 'page'
    page = urllib.request.urlopen(url)
    # parse the html using beautiful soup and store in variable 'soup'
    soup = BeautifulSoup(page, 'html.parser')
    return soup


def get_reviews(soup):
    reviews = []
    ratings = []
    soup_reviews = soup.find_all('div', attrs={'class': 'review-content'})
    for soup_review in soup_reviews:
        # Extract review
        review = soup_review.find('p').text
        if review != FORBIDDEN_TEXT:
            # Extract rating
            rating = soup_review.find('img')['alt'][0]
            # Add review & rating
            reviews.append(review)
            ratings.append(rating)
    return reviews, ratings


def scrap_yelp():
    reviews, ratings = [], []
    url_generator = gen_url_yelp(URL_FORMAT)
    for url in url_generator:
        print(url)
        soup = get_soup(url)
        batch_reviews, batch_ratings = get_reviews(soup)
        reviews += batch_reviews
        ratings += batch_ratings
    return reviews, ratings


if __name__ == '__main__':
    main()
