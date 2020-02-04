from auto_encoder import AutoEncoder

import json
from os import path

import gensim
import nltk
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def word_to_vector(wv, word):
    if word in wv:
        return wv[word]
    else:
        return np.ones((100,))


def texts_to_vectors(wv, texts):
    texts_vectors = np.zeros((len(texts), max(len(text) for text in texts) * wv.vector_size))
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            texts_vectors[i][j * wv.vector_size: j * wv.vector_size + wv.vector_size] = word_to_vector(wv, texts[i][j])
    return np.array(texts_vectors)


def vectors_to_texts(wv, vectors):
    texts = []
    for vector in vectors:
        text = ""
        for i in range(0, len(vector), wv.vector_size):
            if np.sum(vector[i: i + wv.vector_size]) == wv.vector_size:
                text += "<UNK> "
            elif np.sum(vector[i: i + wv.vector_size]) == 0:
                break
            else:
                similar = wv.similar_by_vector(vector[i: i + wv.vector_size])
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

    training_vectors = texts_to_vectors(wv, training)
    validation_vectors = texts_to_vectors(wv, validation)
    test_vectors = texts_to_vectors(wv, test)

    hidden_code_dim = 512
    model = AutoEncoder(len(training_vectors[0]), hidden_code_dim)
    model.train(tf.cast(training_vectors[:4096], dtype=float), tf.cast(validation_vectors[:1024], dtype=float), 12, 64, tf.keras.optimizers.Adam())


if __name__ == "__main__":
    main()
