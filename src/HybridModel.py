import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import LyricsDataExtraction

from CreateDataset import clusters, classes, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE


if __name__ == "__main__":
    # Loading the audio data
    cluster1_dataset = pd.read_json("Cluster 1.json")
    cluster2_dataset = pd.read_json("Cluster 2.json")
    cluster3_dataset = pd.read_json("Cluster 3.json")
    cluster4_dataset = pd.read_json("Cluster 4.json")
    cluster5_dataset = pd.read_json("Cluster 5.json")

    dataset = cluster1_dataset.append(cluster2_dataset)
    dataset = dataset.append(cluster3_dataset)
    dataset = dataset.append(cluster4_dataset)
    dataset = dataset.append(cluster5_dataset)

    dataset = dataset.drop('Class', axis=1)

    print("Read the data from JSON files, created the audio dataframe")

    # Finding the max. frame length
    max_frame_length = 0
    for i in dataset["Mel Spectrogram"]:
        for j in i:
            if len(j) > max_frame_length:
                max_frame_length = len(j)

    # Preprocessing the audio data, so that the frames of all the data is equal
    for i in range(len(dataset["Mel Spectrogram"])):
        for j in range(len(dataset["Mel Spectrogram"].iloc[i])):
            diff = max_frame_length - len(dataset["Mel Spectrogram"].iloc[i][j])
            for x in range(diff):
                dataset["Mel Spectrogram"].iloc[i][j].append(0.0)

    print("Preprocessed the audio data")

    # Loading the lyrical data
    X_lyrics_train, X_lyrics_test, y_train, y_test, filenames_train, filenames_test = LyricsDataExtraction.extract_features_for_hybrid_model(
        "./Lyrics")

    lyrical_feature_size = len(X_lyrics_train.iloc[0])
    X_lyrics_train = X_lyrics_train.values
    X_lyrics_test = X_lyrics_test.values
    print("Created the lyrics dataframe")

    # To process the "Cluster" data, we need to convert its text data to numeric data
    from sklearn.preprocessing import LabelEncoder

    clusters_data_train = y_train
    clusters_data_test = y_test
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(clusters_data_train)
    y_test = encoder.fit_transform(clusters_data_test)

    # 90% train, %10 test data split for the data
    X_audio_train = pd.DataFrame()
    X_audio_test = pd.DataFrame()

    for i in filenames_train:
        X_audio_train = X_audio_train.append(dataset.iloc[int(i) - 1], ignore_index=True)
    for i in filenames_test:
        X_audio_test = X_audio_test.append(dataset.iloc[int(i) - 1], ignore_index=True)

    X_audio_train = X_audio_train.drop(['Cluster', 'Name'], axis=1)
    X_audio_test = X_audio_test.drop(['Cluster', 'Name'], axis=1)

    print("Splitted the data into train and test sets")

    # Reshaping the audio data so that it fits to the CNN
    X_audio_train = [X_audio_train.iloc[i][0] for i in range(len(X_audio_train))]
    X_audio_test = [X_audio_test.iloc[i][0] for i in range(len(X_audio_test))]

    # Reshaping the input size as (input_size, n_mels, max_frame_length, 1) so that it's like an image
    for i in range(len(X_audio_train)):
        for j in range(len(X_audio_train[i])):
            X_audio_train[i][j] = [[X_audio_train[i][j][k]] for k in range(len(X_audio_train[i][j]))]  # Channel size becomes 1

    for i in range(len(X_audio_test)):
        for j in range(len(X_audio_test[i])):
            X_audio_test[i][j] = [[X_audio_test[i][j][k]] for k in range(len(X_audio_test[i][j]))]  # Channel size becomes 1

    from tensorflow import stack
    X_audio_train = stack(X_audio_train)
    X_audio_test = stack(X_audio_test)
    X_lyrics_train = stack(X_lyrics_train)
    X_lyrics_test = stack(X_lyrics_test)
    y_train = stack(y_train)
    y_test = stack(y_test)

    print("Reshaped the data")

    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Dropout, concatenate

    # CNN model for audio data
    model_audio = Sequential([
        Conv2D(32, (3, 3), input_shape=(len(X_audio_train[0]), len(X_audio_train[0][0]), len(X_audio_train[0][0][0])), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding="same"),
        Dropout(0.03375),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding="same"),
        Dropout(0.0675),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding="same"),
        Dropout(0.125),
        Conv2D(192, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 5), strides=(3, 5), padding="same"),
        Dropout(0.25),
        Flatten(),
    ])

    # Multilayer Perceptron model for lyrical data
    model_lyrics = Sequential([
        Dense(64, activation='relu', input_dim=lyrical_feature_size),
        Dense(128, activation='relu'),
        Dropout(0.25),
    ])

    # Merging the audio and lyric models
    merged_models = concatenate([model_audio.output, model_lyrics.output])

    # Hybrid model for both audio and lyrics
    x = Dense(128, activation="relu")(merged_models)
    x = Dense(len(clusters), activation="softmax")(x)

    model_hybrid = Model(inputs=[model_audio.input, model_lyrics.input], outputs=x)

    print("Created the audio and lyric models and merged them into one model")

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers import RMSprop

    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
               name='Adam')  # Old optimizer
    optNew = RMSprop(learning_rate=0.0001)

    model_hybrid.compile(optimizer=optNew,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Compilation done")

    history = model_hybrid.fit(x=[X_audio_train, X_lyrics_train],
                        y=y_train,
                        epochs=100,
                        batch_size=16,
                        validation_data=([X_audio_test, X_lyrics_test], y_test)
                        )

    print("Fitting done")

    from sklearn.metrics import classification_report, confusion_matrix

    print(model_hybrid.evaluate([X_audio_test, X_lyrics_test], y_test))

    predictions = np.argmax(model_hybrid.predict([X_audio_test, X_lyrics_test]), axis=-1)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    model_hybrid.summary()

    finishData = pd.DataFrame(history.history)
    fig, axs = plt.subplots(2)
    axs[0].plot(finishData["accuracy"], label="Train Accuracy")
    axs[0].plot(finishData["val_accuracy"], label="Validation Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_yticks(np.arange(0, 1, 0.1))
    axs[1].plot(finishData["loss"], label="Train Error")
    axs[1].plot(finishData["val_loss"], label="Validation Error")
    axs[1].legend(loc="upper right")
    axs[1].set_yticks(np.arange(2, 0, -0.2))
    plt.show()
