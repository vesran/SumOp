
# Target the restaurant you want to scrap on Yelp
FORBIDDEN_TEXT = "Cet avis a été supprimé pour violation de nos Conditions d'utilisation"
URL_FORMAT = 'https://www.yelp.fr/not_recommended_reviews/howlin-rays-los-angeles-3?not_recommended_start=#'
PATH_TO_SAVE_SCRAPED_COMMENTS = 'data/yelp.txt'

# Embedding model path
PATH_TO_W2V = 'models/w2v_200_30epochs.bin'

# Custom set of stopwords
STOPWORDS = {'the', 'a', 'so', 'an', 'i', 'be', 'of', 'we', 'you', 'they', 'that', 'this', 'my', 'on', 'do', 'as', 'by',
             'will', 'would', 'just', 'about', 'up', 'over', 'what', 'to', 'which', 'where', 'our', 'if', 'at', 'have',
             'has', 'had', 'is', 'how', 'all', 'it', 'for', 'his', 'her', 'its', 'yourself', 'upon', 'some', 'through',
             'he', 'she', 'actually', 'also', 'sometimes', 'esp', 'onto', 'into', 'and', 'there', 'usually', 'in',
             'nyx', 'only', 'probably'}


# Sentiment analysis
PATH_TO_TEST_ANASENT = 'data/anasent/test.sav'
PATH_TO_ANASENT_MODEL = 'models/anaseq/anaseq_002'

# Aspects
ASPECTS = {
    'food': 0.21,
    'service': 0.4,
    'ambience': 0.4
}

ASPECTS_PARAMS = {
    'limit': 100,
}

# Output
PATH_TO_OUTPUT_DATA = 'data/output.csv'
