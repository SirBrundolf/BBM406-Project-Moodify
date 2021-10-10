import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

from CreateDataset import clusters, classes, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE


if __name__ == "__main__":
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

    print("Read the data from JSON files, created the dataframe")

    # Finding the max. frame length
    max_frame_length = 0
    for i in dataset["Mel Spectrogram"]:
        for j in i:
            if len(j) > max_frame_length:
                max_frame_length = len(j)

    # Preprocessing the dataset, so that the frames of all the data is equal
    for i in range(len(dataset["Mel Spectrogram"])):
        for j in range(len(dataset["Mel Spectrogram"].iloc[i])):
            diff = max_frame_length - len(dataset["Mel Spectrogram"].iloc[i][j])
            for x in range(diff):
                dataset["Mel Spectrogram"].iloc[i][j].append(0.0)

    print("Preprocessed the data")

    # To process the "Cluster" data, we need to convert its text data to numeric data
    from sklearn.preprocessing import LabelEncoder

    clusters_data = dataset.iloc[:, 1]
    encoder = LabelEncoder()
    clusters_encoded = encoder.fit_transform(clusters_data)
    dataset['Cluster'] = clusters_encoded

    from sklearn.model_selection import train_test_split

    X = dataset.drop(['Cluster', 'Name'], axis=1)
    y = dataset['Cluster']

    # Reshaping the data so that it fits to the CNN
    new_X = [X.iloc[i][0] for i in range(len(X))]
    new_y = [y.iloc[i].tolist() for i in range(len(y))]

    # Reshaping the input size as (input_size, n_mels, max_frame_length, 1) so that it's like an image
    for i in range(len(new_X)):
        for j in range(len(new_X[i])):
            new_X[i][j] = [[new_X[i][j][k]] for k in range(len(new_X[i][j]))]  # Channel size becomes 1

    # 90% train, %10 test data split
    X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.1)
    #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    print("Splitted the data into train and test sets")

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Dropout

    model = Sequential([
        # Convolutional layer
        Conv2D(32, (3, 3), input_shape=(len(new_X[0]), len(new_X[0][0]), len(new_X[0][0][0])), activation="relu", padding="same"),
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
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same"),
        Dropout(0.5),

        # Fully connected layer
        Flatten(),
        Dense(64, activation="relu"),

        # Output layer
        Dense(len(clusters), activation="softmax")
    ])

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers import RMSprop

    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
               name='Adam')  # Old optimizer
    optNew = RMSprop(learning_rate=0.0001)

    model.compile(optimizer=optNew,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=100,
                        batch_size=128,
                        validation_data=(X_test, y_test)
                        )

    from sklearn.metrics import classification_report, confusion_matrix

    print(model.evaluate(X_test, y_test))

    predictions = np.argmax(model.predict(X_test), axis=-1)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    model.summary()

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
