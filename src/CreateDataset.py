import numpy as np
import pandas as pd
import librosa
import json
import os


clusters = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]
classes = {"Boisterous": "Cluster 1", "Confident": "Cluster 1", "Passionate": "Cluster 1", "Rousing": "Cluster 1", "Rowdy": "Cluster 1",
           "Amiable-good natured": "Cluster 2", "Cheerful": "Cluster 2", "Fun": "Cluster 2", "Rollicking": "Cluster 2", "Sweet": "Cluster 2",
           "Autumnal": "Cluster 3", "Bittersweet": "Cluster 3", "Brooding": "Cluster 3", "Literate": "Cluster 3", "Poignant": "Cluster 3", "Wistful": "Cluster 3",
           "Campy": "Cluster 4", "Humorous": "Cluster 4", "Silly": "Cluster 4", "whimsical": "Cluster 4", "Witty": "Cluster 4", "Wry": "Cluster 4",
           "Agressive": "Cluster 5", "Fiery": "Cluster 5", "Intense": "Cluster 5", "Tense - Anxious": "Cluster 5", "Visceral": "Cluster 5", "Volatile": "Cluster 5"}

FRAME_SIZE = 512
HOP_LENGTH = 256  # Used for overlapping frames
SAMPLE_RATE = 22050

def read_samples(path):
    dataset = {"Name": [],
                "Cluster": [],
                "Class": [],
                "Mel Spectrogram": []}

    for dirpath, dirnames, filenames in os.walk(path):
        if dirpath is not path:   # Ignoring the first path, as it's the genres folder itself with its subfolders
            for i in filenames:
                path = dirpath + '\\' + i
                audio, sr = librosa.load(path, sr=SAMPLE_RATE)

                mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mels=96)

                dataset["Name"].append(path.split("\\")[-1].split(".")[0])
                dataset["Cluster"].append(classes[dirpath.split("\\")[-1]])
                dataset["Class"].append(dirpath.split("\\")[-1])
                dataset["Mel Spectrogram"].append(mel_spectrogram.tolist())

    print(path, "done")
    return dataset


def read_dataset(name, path):
    dataset = read_samples(path)
    print(dataset["Class"])
    dataset_name = str(name + ".json")
    with open(dataset_name, "w") as fp:
        json.dump(dataset, fp)


if __name__ == "__main__":
    read_dataset("Cluster 1", "Audio/Cluster 1")
    read_dataset("Cluster 2", "Audio/Cluster 2")
    read_dataset("Cluster 3", "Audio/Cluster 3")
    read_dataset("Cluster 4", "Audio/Cluster 4")
    read_dataset("Cluster 5", "Audio/Cluster 5")
