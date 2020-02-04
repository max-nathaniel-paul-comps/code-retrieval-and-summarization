import tensorflow as tf
import matplotlib.pyplot as plt
import random


class MultiLayerPerceptron(object):
    def __init__(self, input_dim, output_dim):
        """
        Initialize model parameters based on input and output dimension
        Weights are initialized from a random normal distribution
        Biases are initialized to 0
        As in the the 3Blue1Brown video, there are two hidden layers, each with 16 nodes
        """
        self._params = {"W0": tf.Variable(tf.random.normal((16, input_dim))),
                        "b0": tf.Variable(tf.zeros((16, 1))),
                        "W1": tf.Variable(tf.random.normal((16, 16))),
                        "b1": tf.Variable(tf.zeros((16, 1))),
                        "W2": tf.Variable(tf.random.normal((output_dim, 16))),
                        "b2": tf.Variable(tf.zeros((output_dim, 1)))}

    def predict(self, inputs):
        """
        Predict numbers given input images
        :param inputs: in the form inputs[example][pixeL_in_example]
        :return: predictions in the form predicts[example][predicted_num_one_hot]
        """
        first_hidden_layer = tf.nn.leaky_relu(tf.matmul(self._params["W0"], inputs, transpose_b=True)
                                              + self._params["b0"])
        second_hidden_layer = tf.nn.leaky_relu(tf.matmul(self._params["W1"], first_hidden_layer)
                                               + self._params["b1"])
        predicts = tf.nn.softmax(tf.transpose(tf.matmul(self._params["W2"], second_hidden_layer)
                                              + self._params["b2"]))
        return predicts

    def loss(self, inputs, targets):
        """
        Given a set of inputs with corresponding true outputs ("targets"), calculate the model's current loss
        """
        return tf.reduce_mean(tf.losses.categorical_crossentropy(targets, self.predict(inputs)))

    def _training_step(self, inputs, targets, learning_rate):
        # Calculate the current model loss, using tf.GradientTape() to watch the model parameters
        with tf.GradientTape() as t:
            current_loss = self.loss(inputs, targets)

        # Use the gradient tape's gradient() function to find gradients
        dW0, db0, dW1, db1, dW2, db2 = t.gradient(current_loss, [self._params["W0"], self._params["b0"],
                                                                 self._params["W1"], self._params["b1"],
                                                                 self._params["W2"], self._params["b2"]])

        # Update model parameters based on calculated gradients
        self._params["W0"].assign_sub(learning_rate * dW0)
        self._params["b0"].assign_sub(learning_rate * db0)
        self._params["W1"].assign_sub(learning_rate * dW1)
        self._params["b1"].assign_sub(learning_rate * db1)
        self._params["W2"].assign_sub(learning_rate * dW2)
        self._params["b2"].assign_sub(learning_rate * db2)

    def train(self, inputs, targets, num_epochs, batch_size, learning_rate):
        """
        Train the model!
        :param inputs: the training dataset inputs, in the form inputs[example][pixel]
        :param targets: the training dataset targets, in the form targets[example][true_number_one_hot]
        :param num_epochs: the number of times to iterate over the entire dataset
        :param batch_size: the number of examples to have in each training batch
        :param learning_rate:
        :return:
        """
        for epoch_num in range(1, num_epochs + 1):
            for batch_num in range(int(inputs.shape[0] / batch_size)):
                inputs_batch = inputs[batch_num * batch_size: batch_num * batch_size + batch_size]
                targets_batch = targets[batch_num * batch_size: batch_num * batch_size + batch_size]
                self._training_step(inputs_batch, targets_batch, learning_rate)
            print("Epoch {} of {} completed, loss = {}".format(epoch_num, num_epochs, self.loss(inputs, targets)))


def main():
    assert tf.version.VERSION >= "2.0.0", "TensorFlow 2.0.0 or newer required, %s installed" % tf.version.VERSION
    # Download the MNIST handwriting recognition dataset
    # I said keras isn't gonna be useful for us
    # Here we're just using it to fetch a sample dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Get a random sample image from the test set for later demonstration
    # (We do this now because we're going to flatten the inputs, which the image library can't handle)
    sample_index = random.randrange(x_test.shape[0])
    plt.imshow(x_test[sample_index], cmap='Greys')

    # The neural network wants its inputs to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten input data for use in the neural net
    x_train = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = tf.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    # There are 10 possible values for what digit is shown
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    print("\n{} examples in the training set, {} examples in the test set".format(x_train.shape[0], x_test.shape[0]))

    assert x_train.shape[1] == x_test.shape[1]
    input_dim = x_train.shape[1]
    assert y_train.shape[1] == y_test.shape[1]
    output_dim = y_train.shape[1]
    print("\nInput dimension: {}\nOutput dimension: {}\nJust like in the 3Blue1Brown video!\n".format(input_dim,
                                                                                                      output_dim))

    model = MultiLayerPerceptron(input_dim, output_dim)

    model.train(x_train, y_train, 25, 128, 0.1)

    print("Loss in test set: {}".format(model.loss(x_test, y_test)))

    print("Displaying sample image from test set")
    plt.show()

    print("The neural network thinks the sample image is a {}. It is actually a {}."
          .format(tf.argmax(model.predict([x_test[sample_index]])[0]),
                  tf.argmax(y_test[sample_index])))


if __name__ == "__main__":
    main()
