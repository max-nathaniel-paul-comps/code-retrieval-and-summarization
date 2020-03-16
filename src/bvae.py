import tensorflow as tf
import tensorflow_probability as tfp
import os
import json


def recon_loss_oh_bow(true, pred_prob, vocab_size):
    true_bag_of_words = docs_to_oh_bags_of_words(true, vocab_size)
    recon_all = tf.nn.sigmoid_cross_entropy_with_logits(tf.stop_gradient(true_bag_of_words), pred_prob)
    recon = tf.reduce_mean(tf.reduce_sum(recon_all, axis=-1)) / 10
    return recon


def recon_loss_bow(true, pred_prob, vocab_size):
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


def create_latent_dists(logits, latent_dim):
    means = logits[:, :latent_dim]
    stddevs = tf.math.abs(logits[:, latent_dim:])
    dists = tfp.distributions.Normal(means, stddevs)
    return dists


def docs_to_oh_bags_of_words(docs, vocab_size):
    one_hot = tf.one_hot(docs, vocab_size)
    bags_of_words = tf.reduce_max(one_hot, axis=-2)
    return bags_of_words


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


class RecurrentEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, vocab_size, emb_dim, input_dropout=0.05, name='gru_variational_encoder'):
        super(RecurrentEncoder, self).__init__(
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
        logits = super(RecurrentEncoder, self).call(x, training=training)
        return create_latent_dists(logits, self.latent_dim)


class MlpBowEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, vocab_size, emb_dim, input_dropout=0.05, name='variational_encoder'):
        super(MlpBowEncoder, self).__init__(
            [
                tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=input_dim),
                tf.keras.layers.Dropout(input_dropout, noise_shape=(None, input_dim, 1)),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Activation('tanh'),
                tf.keras.layers.Dense(latent_dim * 2, name='emb_to_hidden'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(latent_dim * 2, name='hidden_to_encoded')
            ],
            name=name
        )
        self.latent_dim = latent_dim

    def call(self, x, training=None, mask=None):
        logits = super(MlpBowEncoder, self).call(x, training=training)
        dists = create_latent_dists(logits, self.latent_dim)
        return dists


class MlpBowNoEmbEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, vocab_size, emb_dim, input_dropout=0.05, name='variational_encoder'):
        super(MlpBowNoEmbEncoder, self).__init__(
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
        logits = super(MlpBowNoEmbEncoder, self).call(x, training=training)
        dists = create_latent_dists(logits, self.latent_dim)
        return dists


class MlpBowDecoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, reconstructed_dim, vocab_size, name='decoder'):
        super(MlpBowDecoder, self).__init__(
            [
                tf.keras.layers.Dense(latent_dim * 2, input_dim=latent_dim, name='encoded_to_hidden'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(vocab_size, name='hidden_to_decoded')
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


encoders = {
    'mlp_bow': MlpBowEncoder,
    'mlp_bow_no_emb': MlpBowNoEmbEncoder,
    'recurrent': RecurrentEncoder
}

decoders = {
    'mlp_bow': MlpBowDecoder,
    'recurrent': RecurrentDecoder
}

recon_losses = {
    'bow': recon_loss_bow,
    'oh_bow': recon_loss_oh_bow,
    'full': recon_loss
}


class BimodalVariationalAutoEncoder(tf.keras.Model):
    def __init__(self, model_path, l_vocab_size, c_vocab_size, tf_name='bvae'):
        super(BimodalVariationalAutoEncoder, self).__init__(name=tf_name)

        if not os.path.isfile(model_path + "model_description.json"):
            raise FileNotFoundError("Model description not found")

        with open(model_path + "model_description.json", 'r') as json_file:
            model_description = json.load(json_file)

        self.l_dim = model_description['l_dim']
        self.c_dim = model_description['c_dim']
        self.l_vocab_size = l_vocab_size
        self.c_vocab_size = c_vocab_size
        self.latent_dim = model_description['latent_dim']
        self.kld_loss_type = model_description['kld_loss_type']

        self.language_encoder = encoders[model_description['l_enc_type']](
            self.l_dim,
            self.latent_dim,
            self.l_vocab_size,
            model_description['l_emb_dim'],
            input_dropout=model_description['input_dropout'],
            name='language_encoder'
        )

        self.source_code_encoder = encoders[model_description['c_enc_type']](
            self.c_dim,
            self.latent_dim,
            self.c_vocab_size,
            model_description['c_emb_dim'],
            input_dropout=model_description['input_dropout'],
            name='source_code_encoder'
        )

        self.language_decoder = decoders[model_description['l_dec_type']](
            self.latent_dim,
            self.l_dim,
            self.l_vocab_size,
            name='language_decoder'
        )

        self.source_code_decoder = decoders[model_description['c_dec_type']](
            self.latent_dim,
            self.c_dim,
            self.c_vocab_size,
            name='source_code_decoder'
        )

        self.recon_loss = recon_losses[model_description['recon_loss_type']]

    def compute_and_add_loss(self, language_batch, source_code_batch, enc_source_code_dists, enc_language_dists,
                             dec_language, dec_source_code,
                             al=0.35, bl=0.15, ac=0.35, bc=0.15):
        if self.kld_loss_type == 'preg':
            language_kld = preg_loss(enc_language_dists, enc_source_code_dists)
            source_code_kld = preg_loss(enc_source_code_dists, enc_language_dists)
        elif self.kld_loss_type == 'mpreg':
            mean_dists = dists_means(enc_language_dists, enc_source_code_dists)
            language_kld = mpreg_loss(enc_language_dists, mean_dists)
            source_code_kld = mpreg_loss(enc_source_code_dists, mean_dists)
        else:
            raise Exception("Invalid KL-divergence loss: %s" % self.kld_loss_type)
        language_recon = self.recon_loss(language_batch, dec_language, self.l_vocab_size)
        source_code_recon = self.recon_loss(source_code_batch, dec_source_code, self.c_vocab_size)
        final_loss = al * language_recon + bl * language_kld + ac * source_code_recon + bc * source_code_kld
        self.add_loss(final_loss)

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
