import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from heapq import nlargest
from sklearn.neighbors import KNeighborsClassifier


# Program driver
if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open(f'{dir_path}/data/tech_crunch_articles.json') as f:
        articles = json.load(f)

    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, stop_words='english')

    x = vectorizer.fit_transform(list(articles.values()))  # each value in the rows of x represents represents TF*IDF

    n_clusters = 3

    # k-means++ initial method to determine cluster centroids
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1, verbose=True)
    km.fit(x)
    print(f'(cluter labels, cluster count): {np.unique(km.labels_, return_counts=True)}')

    articles_aggregate = {}
    # aggregate the text from each cluster
    for i, cluster in enumerate(km.labels_):
        doc = list(articles.values())[i]
        if cluster not in articles_aggregate.keys():
            articles_aggregate[cluster] = doc
        else:
            articles_aggregate[cluster] += doc

    with open(f'{dir_path}/stopwords.txt') as f:
        custom_stopwords = f.read().splitlines()
        _stopwords = set(stopwords.words('english') + list(punctuation) + custom_stopwords)

    keywords, counts = {}, {}
    for cluster in range(n_clusters):
        word_sent = [word for word in word_tokenize(articles_aggregate[cluster].lower()) if word not in _stopwords]
        freq = FreqDist(word_sent)

        # top keywords in each cluster and their count
        keywords[cluster] = nlargest(50, freq, freq.get)
        counts[cluster] = freq  # dict of freq distributions

    # finding unique keys in each cluster
    unique_keys = {}
    for cluster in range(n_clusters):
        keys_other_clusters = []
        for other_cluster in range(n_clusters):
            if other_cluster == cluster:
                continue
            k = keywords[other_cluster]
            keys_other_clusters.extend(k)  # Extend list by appending all the items from the iterable. Equivalent to a[len(a):] = iterable
        xx = keywords[cluster]
        yy = keys_other_clusters
        unique = set(keywords[cluster]) - set(keys_other_clusters)
        unique_keys[cluster] = unique

    print(unique_keys)

    classifier = KNeighborsClassifier()
    classifier.fit(x, km.labels_)  # training phase

    #  "Arya raises $21M to provide farmers in India finance and post-harvest services"
    with open('data/test_article.txt', 'r') as file:
        test_data = file.read().replace('\n', '')

    test = vectorizer.transform([test_data])
    print(f'test article classifed to cluser {classifier.predict(test)}')


