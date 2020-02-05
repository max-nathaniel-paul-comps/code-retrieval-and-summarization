import json
from os import path
import gensim
import nltk
import numpy as np
import os
import tensorflow as tf
from nltk.tokenize import word_tokenize

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
nltk.download('punkt')


def texts_to_tensors(wv, texts):
    text_tensors = np.zeros((len(texts), max(len(text) for text in texts), wv.vector_size))
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            if texts[i][j] in wv:
                text_tensors[i][j] = wv[texts[i][j]]
            else:
                text_tensors[i][j] = np.ones((wv.vector_size,))
    return text_tensors


def tensors_to_texts(wv, tensors):
    texts = []
    for tensor in tensors:
        text = ""
        for i in range(0, len(tensor)):
            if np.sum(tensor[i]) == wv.vector_size:
                text += "<UNK> "
            elif np.sum(tensor[i]) == 0:
                break
            else:
                similar = wv.similar_by_vector(tensor[i])
                text += similar[0][0] + " "
        texts.append(text)
    return texts


def main():
    assert tf.version.VERSION >= "2.0.0", "TensorFlow 2.0.0 or newer required, %s installed" % tf.version.VERSION

    # Load the collection of Reddit jokes (downloaded from https://github.com/taivop/joke-dataset)
    data = json.load(open("reddit_jokes.json"))
    jokes = [item["title"] + " " + item["body"] for item in data]

    print("Tokenizing...")
    jokes = [word_tokenize(joke) for joke in jokes]

    jokes = [joke for joke in jokes if len(joke) < 60]

    training = jokes[: int(6 * len(jokes) / 8)]
    validation = jokes[int(6 * len(jokes) / 8): int(7 * len(jokes) / 8)]
    test = jokes[int(7 * len(jokes) / 8):]

    print("Size of training set: {}, validation: {}, test: {}".format(len(training), len(validation), len(test)))

    print("Creating Word2Vec Embeddings...")

    embeddings_file_name = "embeddings.kv"
    if not path.exists(embeddings_file_name):
        wv = gensim.models.Word2Vec(training).wv
        wv.save(embeddings_file_name)
    wv = gensim.models.KeyedVectors.load(embeddings_file_name, mmap='r')

    training_tensors = texts_to_tensors(wv, training)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, activation='relu',
                             input_shape=(training_tensors.shape[1], training_tensors.shape[2])),
        tf.keras.layers.RepeatVector(training_tensors.shape[1]),
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(training_tensors.shape[2]))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

    model.fit(training_tensors, training_tensors, epochs=50, verbose=1)


if __name__ == "__main__":
    main()
