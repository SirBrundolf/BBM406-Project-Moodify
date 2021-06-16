import os
import sys
import string
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

cluster_types = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]
map_to_clusters = {"Boisterous": "Cluster 1", "Confident": "Cluster 1", "Passionate": "Cluster 1",
                   "Rousing": "Cluster 1", "Rowdy": "Cluster 1",
                   "Amiable-good natured": "Cluster 2", "Cheerful": "Cluster 2", "Fun": "Cluster 2",
                   "Rollicking": "Cluster 2", "Sweet": "Cluster 2",
                   "Autumnal": "Cluster 3", "Bittersweet": "Cluster 3", "Brooding": "Cluster 3",
                   "Literate": "Cluster 3", "Poignant": "Cluster 3", "Wistful": "Cluster 3",
                   "Campy": "Cluster 4", "Humorous": "Cluster 4", "Silly": "Cluster 4", "whimsical": "Cluster 4",
                   "Witty": "Cluster 4", "Wry": "Cluster 4",
                   "Agressive": "Cluster 5", "Fiery": "Cluster 5", "Intense": "Cluster 5",
                   "Tense - Anxious": "Cluster 5", "Visceral": "Cluster 5", "Volatile": "Cluster 5"}
class_types = list(map_to_clusters.keys())


# Extract label from filepath
def extract_label(filepath):
    split_filepath = filepath.split("/")
    lyrics_cluster = split_filepath[-3]
    lyrics_class = split_filepath[-2]
    name = split_filepath[-1][:-4]
    file_num = int(name)
    name = str(file_num)
    if lyrics_cluster not in cluster_types or lyrics_class not in class_types:
        sys.exit("Wrong label for clusters : " + filepath)
    return lyrics_cluster, lyrics_class, name


# Preprocess lyrics converting them to lowercase, removing punctuation, numbers and stopwords
# Also implement stemming or lemmatization considering parameters.
def preprocess(directory_path, clustering, stemming, lemmatization):
    lyrics = []
    clusters = []
    classes = []
    lyrics_names = []
    for subdir, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = subdir + os.sep + filename
                # Extract label from filepath
                lyrics_cluster, lyrics_class, lyrics_name = extract_label(filepath)
                clusters.append(lyrics_cluster)
                classes.append(lyrics_class)
                lyrics_names.append(lyrics_name)
                # Store all lines in lyrics as a string
                with open(filepath) as current_song:
                    text = current_song.read().replace('\n', ' ')
                    lyrics.append(text.lower())
    # Convert lyrics to all lower case for each song
    lyrics = [song.strip() for song in lyrics]
    # Strip all punctuation from lyrics of each song
    table = str.maketrans('', '', string.punctuation)
    lyrics = [song.translate(table) for song in lyrics]
    # Remove all numbers in the lyrics since they are not informative
    lyrics = [re.sub(r'\d+', '', song) for song in lyrics]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    lyrics = [[word for word in song.split() if word not in stop_words] for song in lyrics]
    # Apply stemming or lemmatization according to parameters
    if stemming:
        stemmer = PorterStemmer()
        lyrics = [[stemmer.stem(word) for word in song] for song in lyrics]
    elif lemmatization:
        lemmatizer = WordNetLemmatizer()
        lyrics = [[lemmatizer.lemmatize(word) for word in song] for song in lyrics]
    # Join words of each song to have a lyrics list where each element corresponds to one song's lyrics
    lyrics = [" ".join([word for word in song]) for song in lyrics]
    # Create labels according to classes or clusters
    labels = clusters
    if not clustering:
        labels = classes
    return lyrics, labels, lyrics_names


# Walk over all lyrics in the dataset, extract their labels and words considering directory format
def extract_features(directory_path, test_split=0.1, clustering=True, stemming=False, lemmatization=False):
    lyrics, labels, files = preprocess(directory_path, clustering, stemming, lemmatization)
    return train_test_split(lyrics, labels, test_size=test_split, random_state=5, stratify=labels)


# Walk over all lyrics in the dataset, extract their labels and words considering directory format
# Vectorize words and split them into train and test parts. Returns file order in those two parts as well.
def extract_features_for_hybrid_model(directory_path):
    lyrics, labels, files = preprocess(directory_path, clustering=True, stemming=True, lemmatization=False)

    # Vectorize lyrics with optimized parameters
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=True, min_df=0.002)
    vectors = vectorizer.fit_transform(lyrics)
    words = vectorizer.get_feature_names()
    table = pd.DataFrame(data=vectors.toarray(), columns=words)

    # Splits dataset
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(table, labels, files,
                                                                                         test_size=0.1, random_state=5,
                                                                                         stratify=labels)
    return X_train, X_test, y_train, y_test, filenames_train, filenames_test