import tensorflow as tf


class AutoEncoder(object):
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
