import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import os
import json
import numpy as np
from text_data_utils import *


def ret_recon_loss(true, pred_prob, vocab_size):
    true_bag_of_words = docs_to_bags_of_words(true, vocab_size)
    recon_all = tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(true_bag_of_words), pred_prob)
    recon = tf.reduce_mean(recon_all)
    return recon


def recon_loss(true, pred, _):
    mask = tf.logical_not(tf.equal(true, 0))
    recon_all = tf.keras.losses.sparse_categorical_crossentropy(true, pred, from_logits=True)
    recon_all_masked = tf.where(mask, x=recon_all, y=0.0)
    recon = tf.reduce_sum(recon_all_masked) / tf.reduce_sum(tf.cast(mask, 'float32'))
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


def docs_to_bags_of_words(docs, vocab_size):
    one_hot = tf.one_hot(docs, vocab_size)
    bags_of_words = tf.reduce_mean(one_hot, axis=-2)
    return bags_of_words


class BagOfWords(tf.keras.layers.Layer):
    def __init__(self, vocab_size, name='bag_of_words', **kwargs):
        super(BagOfWords, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size

    def call(self, inputs, **kwargs):
        return docs_to_bags_of_words(inputs, self.vocab_size)


class RecurrentVariationalEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, vocab_size, emb_dim, input_dropout=0.05, name='gru_variational_encoder'):
        super(RecurrentVariationalEncoder, self).__init__(
            [
                tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=input_dim),
                tf.keras.layers.Dropout(input_dropout, noise_shape=(None, input_dim, 1)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(latent_dim, return_sequences=False)),
                tf.keras.layers.Dense(latent_dim * 2)
            ],
            name=name
        )
        self.latent_dim = latent_dim

    def call(self, x, training=None, mask=None):
        dist = super(RecurrentVariationalEncoder, self).call(x, training=training)
        mean = dist[:, :self.latent_dim]
        stddev = tf.math.abs(dist[:, self.latent_dim:])
        return tfp.distributions.Normal(mean, stddev)


class VariationalEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, vocab_size, emb_dim, input_dropout=0.05, name='variational_encoder'):
        super(VariationalEncoder, self).__init__(
            [
                tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=input_dim),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Activation('tanh'),
                tf.keras.layers.Dense(latent_dim * 2),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(latent_dim * 2)
            ],
            name=name
        )
        self.latent_dim = latent_dim

    def call(self, x, training=None, mask=None):
        dist = super(VariationalEncoder, self).call(x, training=training)
        mean = dist[:, :self.latent_dim]
        stddev = tf.math.abs(dist[:, self.latent_dim:])
        return tfp.distributions.Normal(mean, stddev)


class VariationalEncoderNoEmbedding(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, vocab_size, emb_dim, input_dropout=0.05, name='variational_encoder'):
        super(VariationalEncoderNoEmbedding, self).__init__(
            [
                BagOfWords(vocab_size),
                tf.keras.layers.Dense(latent_dim * 2),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(latent_dim * 2)
            ],
            name=name
        )
        self.latent_dim = latent_dim

    def call(self, x, training=None, mask=None):
        dist = super(VariationalEncoderNoEmbedding, self).call(x, training=training)
        mean = dist[:, :self.latent_dim]
        stddev = tf.math.abs(dist[:, self.latent_dim:])
        return tfp.distributions.Normal(mean, stddev)


class Decoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, reconstructed_dim, vocab_size, name='decoder'):
        super(Decoder, self).__init__(
            [
                tf.keras.layers.Dense(latent_dim * 2, input_dim=latent_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(vocab_size)
            ],
            name=name
        )


class RecurrentDecoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, reconstructed_dim, vocab_size, name='recurrent_decoder'):
        super(RecurrentDecoder, self).__init__(
            [
                tf.keras.layers.RepeatVector(reconstructed_dim, input_shape=(latent_dim,)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(latent_dim, return_sequences=True)),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
            ],
            name=name
        )


class BimodalVariationalAutoEncoder(tf.keras.Model):
    def __init__(self, language_dim, l_sw_vocab_size, l_emb_dim, source_code_dim, c_sw_vocab_size, c_emb_dim,
                 latent_dim, input_dropout=0.05, architecture='recurrent', name='bvae'):
        super(BimodalVariationalAutoEncoder, self).__init__(name=name)
        if architecture == 'recurrent':
            l_enc_model = RecurrentVariationalEncoder
            c_enc_model = RecurrentVariationalEncoder
            l_dec_model = RecurrentDecoder
            c_dec_model = RecurrentDecoder
            self.recon_loss = recon_loss
        elif architecture == 'mlp':
            l_enc_model = VariationalEncoder
            c_enc_model = VariationalEncoder
            l_dec_model = Decoder
            c_dec_model = Decoder
            self.recon_loss = ret_recon_loss
        elif architecture == 'mlp_parsed_code':
            l_enc_model = VariationalEncoderNoEmbedding
            c_enc_model = VariationalEncoderNoEmbedding
            l_dec_model = Decoder
            c_dec_model = Decoder
            self.recon_loss = ret_recon_loss
        else:
            raise Exception("Invalid architecture specification %s" % architecture)
        self.language_encoder = l_enc_model(language_dim, latent_dim, l_sw_vocab_size, l_emb_dim,
                                            input_dropout=input_dropout, name='language_encoder')
        self.source_code_encoder = c_enc_model(source_code_dim, latent_dim, c_sw_vocab_size, c_emb_dim,
                                               input_dropout=input_dropout, name='source_code_encoder')
        self.language_decoder = l_dec_model(latent_dim, language_dim, l_sw_vocab_size,
                                            name='language_decoder')
        self.source_code_decoder = c_dec_model(latent_dim, source_code_dim, c_sw_vocab_size,
                                               name='source_code_decoder')
        self.l_vocab_size = l_sw_vocab_size
        self.c_vocab_size = c_sw_vocab_size

    def compute_and_add_loss(self, language_batch, source_code_batch, enc_source_code_dists, enc_language_dists,
                             dec_language, dec_source_code):
        mean_dists = dists_means(enc_language_dists, enc_source_code_dists)
        language_kld = mpreg_loss(enc_language_dists, mean_dists)
        source_code_kld = mpreg_loss(enc_source_code_dists, mean_dists)
        """language_kld = preg_loss(enc_language_dists, enc_source_code_dists)
        source_code_kld = preg_loss(enc_source_code_dists, enc_language_dists)"""
        language_recon = self.recon_loss(language_batch, dec_language, self.l_vocab_size)
        source_code_recon = self.recon_loss(source_code_batch, dec_source_code, self.c_vocab_size)
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


def bvae_demo(model_path="../models/a3/", tokenizers_path="../data/iyer_csharp/"):

    language_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizers_path + "language_tokenizer")
    code_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizers_path + "code_tokenizer")

    if not os.path.isfile(model_path + "model_description.json"):
        raise FileNotFoundError("Model description not found")

    with open(model_path + "model_description.json", 'r') as json_file:
        model_description = json.load(json_file)

    l_dim = model_description['l_dim']
    l_emb_dim = model_description['l_emb_dim']
    c_dim = model_description['c_dim']
    c_emb_dim = model_description['c_emb_dim']
    latent_dim = model_description['latent_dim']
    dropout_rate = model_description['dropout_rate']
    architecture = model_description['architecture']

    model = BimodalVariationalAutoEncoder(l_dim, language_tokenizer.vocab_size, l_emb_dim,
                                          c_dim, code_tokenizer.vocab_size, c_emb_dim,
                                          latent_dim, input_dropout=dropout_rate, architecture=architecture)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=False)

    model.load_weights(model_path + "model_checkpoint.ckpt")

    while True:
        summary = input("Input Summary: ")
        if summary == "exit":
            quit(0)
        summary = [language_tokenizer.encode(preprocess_language(summary))]
        summary = pad_sequences(summary, maxlen=l_dim, padding='post', value=0)
        latent = model.language_encoder(summary).mean()
        source_code = model.source_code_decoder(latent)[0].numpy()
        source_code = code_tokenizer.decode(np.argmax(source_code, axis=-1))
        print("Generated Source Code: ", source_code)


if __name__ == "__main__":
    bvae_demo()
