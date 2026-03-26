import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = [
    "Data science and machine learning",
    "Deep learning techniques in AI",
    "Natural language processing applications",
    "Football and cricket are popular sports",
    "Basketball is an exciting game",
    "Sports events attract many fans"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(text_data)

k = 2

kmeans = KMeans(n_clusters=k)
kmeans.fit(tfidf_matrix)


clusters = kmeans.predict(tfidf_matrix)

for i in range(len(text_data)):
    print("Text:", text_data[i])
    print("Cluster:", clusters[i])
    print()