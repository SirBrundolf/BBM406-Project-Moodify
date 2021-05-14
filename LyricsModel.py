import numpy as np
import pandas as pd
import os

from CreateDataset import clusters, classes

if __name__ == "__main__":
    def read_lyrics(file_name):
        file = open(file_name, "r", encoding="utf8")
        comments = file.readlines()
        return comments


    def read_samples(path):
        dataset = {"Cluster": [],
                   "Class": [],
                   "Text": []}

        for dirpath, dirnames, filenames in os.walk(path):
            if dirpath is not path:  # Ignoring the first path, as it's the genres folder itself with its subfolders
                for i in filenames:
                    path = dirpath + '\\' + i

                    dataset["Cluster"].append(classes[dirpath.split("\\")[-1]])
                    dataset["Class"].append(dirpath.split("\\")[-1])
                    dataset["Text"].append(read_lyrics(path))

        print(path, "done")
        return dataset

    cluster1_dataset = pd.DataFrame(read_samples("Lyrics/Cluster 1"))
    cluster2_dataset = pd.DataFrame(read_samples("Lyrics/Cluster 2"))
    cluster3_dataset = pd.DataFrame(read_samples("Lyrics/Cluster 3"))
    cluster4_dataset = pd.DataFrame(read_samples("Lyrics/Cluster 4"))
    cluster5_dataset = pd.DataFrame(read_samples("Lyrics/Cluster 5"))

    dataset = cluster1_dataset.append(cluster2_dataset)
    dataset = dataset.append(cluster3_dataset)
    dataset = dataset.append(cluster4_dataset)
    dataset = dataset.append(cluster5_dataset)

    print(dataset)

    from sklearn.model_selection import train_test_split

    X = dataset.drop('Cluster', axis=1)
    y = dataset['Cluster']

    # 80% train, %20 test data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

