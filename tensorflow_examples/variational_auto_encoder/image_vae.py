import tensorflow as tf
import tensorflow_probability as tfp
import random
import matplotlib.pyplot as plt


class MLPVariationalAutoEncoder(object):
    """
    A variational auto-encoder based on multilayer perceptrons
    """
    def __init__(self, input_dim, code_dim):
        """
        Initialize model parameters based on input and code dimensions
        This function is exactly the same as the regular auto-encoder
        *except* that the final step of the encoder generates a mean and stddev instead of single values
        So the dimensionality of the final step of the encoder is multiplied by 2
        """
        assert code_dim < input_dim
        hidden_1_dim = int(input_dim - (input_dim - code_dim) / 4)
        hidden_2_dim = int(input_dim - 3 * (input_dim - code_dim) / 4)
        self._params = {
            "enc_W0": tf.Variable(tf.random.normal((hidden_1_dim, input_dim), mean=0.0, stddev=0.05)),
            "enc_b0": tf.Variable(tf.zeros((hidden_1_dim, 1))),
            "enc_W1": tf.Variable(tf.random.normal((hidden_2_dim, hidden_1_dim), mean=0.0, stddev=0.05)),
            "enc_b1": tf.Variable(tf.zeros((hidden_2_dim, 1))),
            "enc_W2": tf.Variable(tf.random.normal((code_dim * 2, hidden_2_dim), mean=0.0, stddev=0.05)),
            "enc_b2": tf.Variable(tf.zeros((code_dim * 2, 1))),
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
        encoded_distribution = tf.transpose(tf.matmul(self._params["enc_W2"], hidden_layer_2)
                                            + self._params["enc_b2"])
        code_dim = int(encoded_distribution.shape[1] / 2)
        mean = encoded_distribution[:, :code_dim]
        stddev = encoded_distribution[:, code_dim:] ** 2
        return tfp.distributions.Normal(mean, stddev)

    def decode(self, sampled_encoding):
        hidden_layer_1 = tf.nn.leaky_relu(tf.matmul(self._params["dec_W0"], sampled_encoding, transpose_b=True)
                                          + self._params["dec_b0"])
        hidden_layer_2 = tf.nn.leaky_relu(tf.matmul(self._params["dec_W1"], hidden_layer_1)
                                          + self._params["dec_b1"])
        decoded = tf.transpose(tf.nn.sigmoid(tf.matmul(self._params["dec_W2"], hidden_layer_2)
                                             + self._params["dec_b2"]))
        return decoded

    def loss(self, inputs):
        encoded = self.encode(inputs)
        kl_divergences = tfp.distributions.kl_divergence(encoded,
                                                         tfp.distributions.Normal(0.0, 1.0))
        kl_divergence = tf.reduce_mean(tf.reduce_mean(kl_divergences, axis=1))
        sample_encoded = encoded.sample()
        decoded = self.decode(sample_encoded)
        mean_square_error = tf.reduce_mean(tf.losses.mean_squared_error(inputs, decoded))
        return mean_square_error + 0.01 * kl_divergence

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


def main():
    assert tf.version.VERSION >= "2.0.0", "TensorFlow 2.0.0 or newer required, %s installed" % tf.version.VERSION

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print("\n{} examples in the training set, {} examples in the test set".format(x_train.shape[0], x_test.shape[0]))

    assert x_train.shape[1] == x_test.shape[1]
    assert x_train.shape[2] == x_test.shape[2]
    input_x_dim = x_train.shape[1]
    input_y_dim = x_train.shape[2]
    print("\nInput dimension: {}x{}\n".format(input_x_dim, input_y_dim))

    input_dim = input_x_dim * input_y_dim
    x_train = tf.reshape(x_train, (len(x_train), input_dim))
    x_test = tf.reshape(x_test, (len(x_test), input_dim))

    hidden_code_dim = 36
    model = MLPVariationalAutoEncoder(input_dim, hidden_code_dim)
    model.train(x_train, x_test, 30, 1024, tf.keras.optimizers.Adam(learning_rate=0.0001))

    for _ in range(5):
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        test_case = x_test[random.randrange(x_test.shape[0])]
        test_case_img = tf.reshape(test_case, (1, input_x_dim, input_y_dim))[0] * 255.0
        plt.imshow(test_case_img, cmap='Greys')

        plt.subplot(1, 3, 2)
        plt.title("Hidden Representation")
        encoded_dist = model.encode([test_case])
        encoded = encoded_dist.sample()
        # The reshape command makes the 16-long hidden code by 4x4
        # so we can see it alongside the input and output
        encoded_img = tf.reshape(tf.nn.sigmoid(encoded), (1, 6, 6))[0] * 255.0
        plt.imshow(encoded_img, cmap='Greys')

        plt.subplot(1, 3, 3)
        plt.title("Output Image")
        decoded = model.decode([encoded])
        decoded_img = tf.reshape(decoded, (1, input_x_dim, input_y_dim))[0] * 255.0
        plt.imshow(decoded_img, cmap='Greys')

        plt.show()


if __name__ == "__main__":
    main()
