# BBM406-Project-Moodify
Me and my team's course project for BBM406 - Moodify.

A music mood classifier which has 3 different models (utilizing audio data with CNNs, lyrical data with machine learning methods and both with a hybrid model using early fusion).

## Installation
Download Multi-modal MIREX-like emotion dataset (2013) from http://mir.dei.uc.pt/downloads.html

Extract the Audio and Lyrics files and split them into subfolders using the .bat files in the zip.

Make sure that the files are splitted as: Audio/Cluster (Count)/(Class Name) and Lyrics/Cluster (Count)/(Class Name)

After the splitting is done, run CreateDataset.py to extract the audio dataset as .json files.

To run a model, run their respective .py files (AudioModel.py for audio, LyricsModels.py for lyrics, HybridModel.py for hybrid).
