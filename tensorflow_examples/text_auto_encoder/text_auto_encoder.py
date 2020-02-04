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


class TextAutoEncoder(object):
    def __init__(self, input_dim, code_dim):
        """
        Initialize model parameters based on input and code dimensions
        The code dim should be much smaller than the input dim to demonstrate the model's encoding abilities
        The encoder has two hidden layers, and so does the decoder
        These hidden layers go down in size as you get to the hidden state
        """
        assert code_dim < input_dim, "Are you trying to make me learn the identity function or something?"
        hidden_1_dim = int(input_dim - (input_dim - code_dim) / 4)
        hidden_2_dim = int(input_dim - 3 * (input_dim - code_dim) / 4)
        self._params = {
            "enc_W0": tf.Variable(tf.random.normal((hidden_1_dim, input_dim), mean=0.0, stddev=0.05)),
            "enc_b0": tf.Variable(tf.zeros((hidden_1_dim, 1))),
            "enc_W1": tf.Variable(tf.random.normal((hidden_2_dim, hidden_1_dim), mean=0.0, stddev=0.05)),
            "enc_b1": tf.Variable(tf.zeros((hidden_2_dim, 1))),
            "enc_W2": tf.Variable(tf.random.normal((code_dim, hidden_2_dim), mean=0.0, stddev=0.05)),
            "enc_b2": tf.Variable(tf.zeros((code_dim, 1))),
            "dec_W0": tf.Variable(tf.random.normal((hidden_2_dim, code_dim), mean=0.0, stddev=0.05)),
            "dec_b0": tf.Variable(tf.zeros((hidden_2_dim, 1))),
            "dec_W1": tf.Variable(tf.random.normal((hidden_1_dim, hidden_2_dim), mean=0.0, stddev=0.05)),
            "dec_b1": tf.Variable(tf.zeros((hidden_1_dim, 1))),
            "dec_W2": tf.Variable(tf.random.normal((input_dim, hidden_1_dim), mean=0.0, stddev=0.05)),
            "dec_b2": tf.Variable(tf.zeros((input_dim, 1))),
        }

    def encode(self, inputs):
        hidden_layer_1 = tf.nn.leaky_relu(tf.matmul(self._params["enc_W0"], inputs, transpose_b=True)
                                          + self._params["enc_b0"])
        hidden_layer_2 = tf.nn.leaky_relu(tf.matmul(self._params["enc_W1"], hidden_layer_1)
                                          + self._params["enc_b1"])
        encoded = tf.transpose(tf.matmul(self._params["enc_W2"], hidden_layer_2)
                               + self._params["enc_b2"])
        return encoded

    def decode(self, encoded):
        hidden_layer_1 = tf.nn.leaky_relu(tf.matmul(self._params["dec_W0"], encoded, transpose_b=True)
                                          + self._params["dec_b0"])
        hidden_layer_2 = tf.nn.leaky_relu(tf.matmul(self._params["dec_W1"], hidden_layer_1)
                                          + self._params["dec_b1"])
        decoded = tf.transpose(tf.nn.sigmoid(tf.matmul(self._params["dec_W2"], hidden_layer_2)
                                             + self._params["dec_b2"]))
        return decoded

    def loss(self, inputs):
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        mean_square_error = tf.reduce_mean(tf.losses.mean_squared_error(inputs, decoded))
        return mean_square_error

    def _training_step(self, inputs, optimizer):
        with tf.GradientTape() as t:
            current_loss = self.loss(inputs)
        grads = t.gradient(current_loss, self._params)
        optimizer.apply_gradients((grads[key], self._params[key]) for key in grads.keys())

    def train(self, inputs, val_inputs, num_epochs, batch_size, optimizer):
        for epoch_num in range(1, num_epochs + 1):
            for batch_num in range(int(inputs.shape[0] / batch_size)):
                inputs_batch = inputs[batch_num * batch_size: batch_num * batch_size + batch_size]
                self._training_step(inputs_batch, optimizer)
            train_loss = self.loss(inputs)
            val_loss = self.loss(val_inputs)
            print("Epoch {} of {} completed, training loss = {}, validation loss = {}".format(
                epoch_num, num_epochs, train_loss, val_loss))
            if val_loss - train_loss > train_loss / 32:
                print("Stopped because validation loss significantly exceeds training loss")
                break


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
    model = TextAutoEncoder(len(training_vectors[0]), hidden_code_dim)
    model.train(tf.cast(training_vectors[:4096], dtype=float), tf.cast(validation_vectors[:1024], dtype=float), 12, 64, tf.keras.optimizers.Adam())


if __name__ == "__main__":
    main()
