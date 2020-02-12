import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random
import matplotlib.pyplot as plt


class MultiLayerPerceptron(tf.keras.layers.Layer):
    def __init__(self, from_dim, to_dim, hidden_layer_dims, final_activation, name='multi_layer_perceptron', **kwargs):
        super(MultiLayerPerceptron, self).__init__(name=name, **kwargs)
        self._layers = list()
        prev_dim = from_dim
        for hidden_layer_dim in hidden_layer_dims:
            self._layers.append(tf.keras.layers.Dense(hidden_layer_dim, activation=None, input_dim=prev_dim))
            self._layers.append(tf.keras.layers.LeakyReLU())
            prev_dim = hidden_layer_dim
        self._layers.append(tf.keras.layers.Dense(to_dim, input_dim=prev_dim, activation=final_activation))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


class MLPVariationalAutoEncoder(tf.keras.models.Model):
    def __init__(self, input_dim, latent_dim, hidden_layer_dims: list, final_activation='sigmoid',
                 name='variational_auto_encoder', **kwargs):
        super(MLPVariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._encoder = MultiLayerPerceptron(input_dim, latent_dim * 2, hidden_layer_dims, None, name='encoder')
        self._decoder = MultiLayerPerceptron(latent_dim, input_dim, reversed(hidden_layer_dims), final_activation,
                                             name='decoder')

    def encode(self, inputs):
        latent_dist_raw = self._encoder(inputs)
        latent_dist = tfp.distributions.Normal(latent_dist_raw[:, :self._latent_dim],
                                               latent_dist_raw[:, self._latent_dim:])
        return latent_dist

    def decode(self, latent):
        return self._decoder(latent)

    def call(self, inputs, training=None, mask=None):
        latent_dist = self.encode(inputs)
        latent = latent_dist.sample()
        emp_dist = tfp.distributions.Empirical(tf.transpose(latent))
        kl_divergence = tf.reduce_mean(
            tfp.distributions.kl_divergence(
                tfp.distributions.Normal(emp_dist.mean(), emp_dist.stddev()),
                tfp.distributions.Normal(0.0, 1.0)
            )
        )
        self.add_loss(kl_divergence)

        decoded = self.decode(latent)
        return decoded


def main():
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
    x_train = np.reshape(x_train, (len(x_train), input_dim))
    x_test = np.reshape(x_test, (len(x_test), input_dim))

    hidden_code_dim = 16
    model = MLPVariationalAutoEncoder(input_dim, hidden_code_dim, [512, 128])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.mean_squared_error)

    model.fit(x_train, x_train, batch_size=512, epochs=25, verbose=1, validation_data=(x_test, x_test), shuffle=True)

    for _ in range(4):
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        test_case = x_test[random.randrange(x_test.shape[0])]
        test_case_img = tf.reshape(test_case, (1, input_x_dim, input_y_dim))[0] * 255.0
        plt.imshow(test_case_img, cmap='Greys')

        plt.subplot(1, 3, 2)
        plt.title("Hidden Representation")
        encoded_dist = model.encode(np.array([test_case]))
        encoded = encoded_dist.sample()
        # The reshape command makes the 16-long hidden code by 4x4
        # so we can see it alongside the input and output
        encoded_img = tf.reshape(tf.nn.sigmoid(encoded), (1, 4, 4))[0] * 255.0
        plt.imshow(encoded_img, cmap='Greys')

        plt.subplot(1, 3, 3)
        plt.title("Output Image")
        decoded = model.decode(np.array([encoded]))
        decoded_img = tf.reshape(decoded, (1, input_x_dim, input_y_dim))[0] * 255.0
        plt.imshow(decoded_img, cmap='Greys')

        plt.show()


if __name__ == "__main__":
    main()
