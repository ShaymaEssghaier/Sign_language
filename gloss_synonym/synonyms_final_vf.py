# -*- coding: utf-8 -*-
"""synonyms_final_vf.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HT_gO14OTopHWuccQ5bzYWFn5KpUZxhi
"""




import pandas as pd
import spacy
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet

def load_data(file_path):
    """
    This function loads the data from a given file_path

    parameter: str the file path

    Returns: the unique words in gloss column
    """
    data = pd.read_csv(file_path, delimiter=";")
    return data["gloss"].unique()

def initialize_spacy_model(model_name="en_core_web_md"):
    return spacy.load(model_name)

def download_wordnet():
    """
    This function downloads a dictionary that will be used to find antonyms
    """
    nltk.download('wordnet')

def generate_word_vectors(words, model):
    return np.array([model(word).vector for word in words])

def plot_k_distance_graph(distances, k):
    k_distances = np.sort(distances, axis=1)[:, k]
    k_distances = np.sort(k_distances)
    plt.figure(figsize=(10, 5))
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.title(f'k-distance Graph for k={k}')
    plt.grid(True)
    plt.show()

def perform_dbscan_clustering(word_vectors, eps, min_samples=5):
    dbscan = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)
    dbscan.fit(word_vectors)
    return dbscan

def create_cluster_mapping(words, dbscan_labels):
    cluster_to_words = {}
    for word, cluster in zip(words, dbscan_labels):
        if cluster not in cluster_to_words:
            cluster_to_words[cluster] = []
        cluster_to_words[cluster].append(word)
    return cluster_to_words

def find_antonyms(word):
    antonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    return antonyms

def find_synonyms_in_cluster(word, model, cluster_to_words, dbscan_model):
    """
    This function finds the most similar word in the same cluster, and excludes antonyms
    """
    word_vector = model(word).vector
    cluster_label = dbscan_model.fit_predict([word_vector])[0]
    cluster_words = cluster_to_words.get(cluster_label, [])

    if not cluster_words:
        return None

    antonyms = find_antonyms(word)
    similarities = [(dict_word, model(dict_word).similarity(model(word))) for dict_word in cluster_words if dict_word != word and dict_word not in antonyms]

    if not similarities:
        return None

    most_similar_word = sorted(similarities, key=lambda item: -item[1])[0][0]
    return most_similar_word

def display_clusters(cluster_to_words):
    for cluster_label, words in cluster_to_words.items():
        if cluster_label != -1:  # Exclude noise points
            print(f"Cluster {cluster_label}: {words}")
        else:
            print(f"Noise: {words}")

def main(file_path, model_name="en_core_web_md", eps=0.23, min_samples=5, k=5):
    global nlp, cluster_to_words, dbscan

    dict_2000 = load_data(file_path)
    nlp = initialize_spacy_model(model_name)
    download_wordnet()

    word_vectors = generate_word_vectors(dict_2000, nlp)

    # distances = cosine_distances(word_vectors)
    # plot_k_distance_graph(distances, k)

    dbscan = perform_dbscan_clustering(word_vectors, eps, min_samples)
    cluster_to_words = create_cluster_mapping(dict_2000, dbscan.labels_)

if __name__ == "__main__":
    main("filtered_WLASL.csv")

##TEST##
target_word = "unhappy"
synonym = find_synonyms_in_cluster(target_word, nlp, cluster_to_words, dbscan)
print(f"The most similar word to '{target_word}' is '{synonym}'")

##If you want to see clusters##
num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
print(f"Number of clusters: {num_clusters}")

cluster_label = dbscan.fit_predict([nlp("unhappy").vector])[0]
same_cluster_words = cluster_to_words.get(cluster_label, [])
print(f"Words in the same cluster as 'unhappy': {same_cluster_words}")

display_clusters(cluster_to_words)

