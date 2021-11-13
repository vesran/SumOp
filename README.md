# SumOp

Summarising the opinion of a batch of reviews by producing an instant views showing topics and sentiments.

Provides a visual summary of a batch of reviews. Given the url of a restaurant, 
this module scraps a list of reviews and estimates the sentiment. Each sentence is cut 
to isolate a sentiment in order to handle multi-opinionated texts. Aspects/Topics are detected within 
each fragment. 

# TLDR
One main assumption can be underlined after analysing multiple datasets : 
sentiments are usually grouped within a sentence (ie a sentence starts by positive statements and 
ends with a negative review).
In order to estimate the sentiment within each aspect, a NN model is trained to 
estimate the polarity of each word in a sentence. It allows us to segment sentences 
with multiples sentiments. Each fragment of text may several aspects but one sentiment. 
From now, it's only a matter of detecting aspects.

# Pipeline

![pipeline](imgs/pipeline.png)


## Sentiment analysis

![Exemple sentiment analysis](imgs/ex_anasent.png)

## Aspect detection

Used a modiified version of the aspect detection procedure by Tulken and Cranenburgh (2020).

![aspect detection](imgs/aspect_detection.png)


# TODOs

- [X] Code
- [X] Add anasent notebook
- [ ] Dashboard
- [ ] Docker
- [ ] Add pdf
- [ ] Explain sentiment analysis part
- [ ] Explain aspect detection part


# References
* <a href="https://arxiv.org/abs/2004.13580">Embarrassingly Simple Unsupervised Aspect Extraction</a>, 2020