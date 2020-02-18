import tensorflow_probability as tfp
from mlp import *


class VariationalEncoder(MultiLayerPerceptron):
    def __init__(self, input_dim, latent_dim, name='variational_encoder'):
        hidden_1_dim = int(input_dim - (input_dim - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(VariationalEncoder, self).__init__(input_dim, 2 * latent_dim, [hidden_1_dim, hidden_2_dim], name=name)
        self.latent_dim = latent_dim

    def __call__(self, x):
        dist = super(VariationalEncoder, self).__call__(x)
        mean = dist[:, :self.latent_dim]
        stddev = tf.math.abs(dist[:, self.latent_dim:])
        return tfp.distributions.Normal(mean, stddev)


class Decoder(MultiLayerPerceptron):
    def __init__(self, latent_dim, reconstructed_dim, name='decoder'):
        hidden_1_dim = int(reconstructed_dim - (reconstructed_dim - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(Decoder, self).__init__(latent_dim, reconstructed_dim, [hidden_2_dim, hidden_1_dim], name=name)
