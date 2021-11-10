
# Target the restaurant you want to scrap on Yelp
FORBIDDEN_TEXT = "Cet avis a été supprimé pour violation de nos Conditions d'utilisation"
URL_FORMAT = 'https://www.yelp.fr/not_recommended_reviews/howlin-rays-los-angeles-3?not_recommended_start=#'
PATH_TO_SAVE_SCRAPED_COMMENTS = 'data/validation/yelp.txt'


# Cleaning texts for sentiment segmentation
STOPWORDS = {'the', 'a', 'so', 'an', 'i', 'be', 'of', 'we', 'you', 'they', 'that', 'this', 'my', 'on', 'do', 'as', 'by',
             'will', 'would',
             'just', 'about', 'up', 'over', 'what', 'to', 'which', 'where', 'our', 'if', 'at', 'have', 'has', 'had',
             'is',
             'how', 'all', 'it', 'for', 'his', 'her', 'its', 'yourself', 'upon', 'some', 'through', 'he', 'she',
             'actually',
             'also', 'sometimes', 'esp', 'onto', 'into', 'and', 'there', 'usually', 'in', 'nyx', 'only', 'probably'}


# Sentiment analysis
PATH_TO_TEST_ANASENT = 'data/anasent/test.sav'
PATH_TO_ANASENT_MODEL = 'embedders/anaseq/anaseq_002'
