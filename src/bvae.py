import tensorflow as tf
import tensorflow_probability as tfp
import random
import os
import json
from text_data_utils import *


def recon_loss(true, pred):
    mask = tf.reduce_all(tf.logical_not(tf.equal(true, 0.0)), axis=-1)
    recon_tensor = tf.losses.cosine_similarity(true, pred) + 1
    recon_masked = tf.where(mask, x=recon_tensor, y=0.0)
    recon = tf.reduce_sum(recon_masked) / tf.reduce_sum(tf.cast(mask, 'float32'))
    return recon


def preg_loss(dists_a, dists_b):
    kl_divergence = tf.reduce_mean(
        tfp.distributions.kl_divergence(
            dists_a,
            dists_b
        )
    )
    logged_kld = tf.math.log(kl_divergence + 1)
    return logged_kld


def dists_means(dists_a, dists_b):
    mean_mean = (dists_a.mean() + dists_b.mean()) / 2
    mean_stddev = (dists_a.stddev() + dists_b.stddev()) / 2
    mean_dists = tfp.distributions.Normal(mean_mean, mean_stddev)
    return mean_dists


def mpreg_loss(dists, mean_dist):
    kl_divergence = tf.reduce_mean(
        tfp.distributions.kl_divergence(
            dists,
            mean_dist
        )
    )
    return kl_divergence


class VariationalEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, wv_size, input_dropout=0.05, name='variational_encoder'):
        hidden_1_dim = int(input_dim * wv_size - (input_dim * wv_size - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(VariationalEncoder, self).__init__(
            [
                tf.keras.layers.Input(shape=(input_dim, wv_size)),
                tf.keras.layers.Dropout(input_dropout, noise_shape=(None, input_dim, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(hidden_1_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(hidden_2_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(latent_dim * 2)
            ],
            name=name
        )
        self.latent_dim = latent_dim

    def call(self, x, training=None, **kwargs):
        dist = super(VariationalEncoder, self).call(x, training=training, **kwargs)
        mean = dist[:, :self.latent_dim]
        stddev = tf.math.abs(dist[:, self.latent_dim:])
        return tfp.distributions.Normal(mean, stddev)


class Decoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, reconstructed_dim, wv_size, name='decoder'):
        hidden_1_dim = int(reconstructed_dim * wv_size - (reconstructed_dim * wv_size - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(Decoder, self).__init__(
            [
                tf.keras.layers.Dense(hidden_2_dim, input_dim=latent_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(hidden_1_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(reconstructed_dim * wv_size),
                tf.keras.layers.Reshape((reconstructed_dim, wv_size))
            ],
            name=name
        )


class BimodalVariationalAutoEncoder(tf.keras.Model):
    def __init__(self, language_dim, source_code_dim, latent_dim, wv_size, input_dropout=0.05, name='bvae'):
        super(BimodalVariationalAutoEncoder, self).__init__(name=name)
        self.language_encoder = VariationalEncoder(language_dim, latent_dim, wv_size, input_dropout=input_dropout,
                                                   name='language_encoder')
        self.source_code_encoder = VariationalEncoder(source_code_dim, latent_dim, wv_size, input_dropout=input_dropout,
                                                      name='source_code_encoder')
        self.language_decoder = Decoder(latent_dim, language_dim, wv_size, name='language_decoder')
        self.source_code_decoder = Decoder(latent_dim, source_code_dim, wv_size, name='source_code_decoder')

    def compute_and_add_loss(self, language_batch, source_code_batch, enc_source_code_dists, enc_language_dists,
                             dec_language, dec_source_code):
        mean_dists = dists_means(enc_language_dists, enc_source_code_dists)
        language_kld = mpreg_loss(enc_language_dists, mean_dists)
        source_code_kld = mpreg_loss(enc_source_code_dists, mean_dists)
        """language_kld = preg_loss(enc_language_dists, enc_source_code_dists)
        source_code_kld = preg_loss(enc_source_code_dists, enc_language_dists)"""
        language_recon = recon_loss(language_batch, dec_language)
        source_code_recon = recon_loss(source_code_batch, dec_source_code)
        self.add_loss(language_kld + source_code_kld + language_recon + source_code_recon)

    def call(self, inputs, training=None, mask=None):
        language_batch = inputs[0]
        source_code_batch = inputs[1]
        enc_language_dists = self.language_encoder(language_batch, training=training)
        enc_source_code_dists = self.source_code_encoder(source_code_batch, training=training)
        enc_language = enc_language_dists.sample()
        enc_source_code = enc_source_code_dists.sample()
        dec_language = self.language_decoder(enc_language, training=training)
        dec_source_code = self.source_code_decoder(enc_source_code, training=training)
        self.compute_and_add_loss(language_batch, source_code_batch, enc_source_code_dists, enc_language_dists,
                                  dec_language, dec_source_code)
        return dec_language, dec_source_code


def main():
    if not os.path.isdir("saved_model"):
        print("Error: Saved model does not exist. Create it with train_model.py")
        quit(-1)

    language_wv = gensim.models.KeyedVectors.load("saved_model/language_wv.txt")
    code_wv = gensim.models.KeyedVectors.load("saved_model/code_wv.txt")

    with open("saved_model/model_description.json", 'r') as json_file:
        model_description = json.load(json_file)

    model = BimodalVariationalAutoEncoder(model_description['language_dim'], model_description['source_code_dim'],
                                          model_description['latent_dim'], model_description['wv_size'])
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.load_weights("saved_model/model_weights")

    while True:
        summary = input("Input Summary: ")
        if summary == "exit":
            quit(0)
        summary = tokenize_text(summary)
        summary = tokenized_texts_to_tensor([summary], language_wv, model_description['language_dim'])
        latent = model.language_encoder(summary).mean()
        source_code = model.source_code_decoder(latent).numpy()
        source_code = tensor_to_tokenized_texts(source_code, code_wv)[0]
        print("Generated Source Code: ", source_code)


if __name__ == "__main__":
    main()
