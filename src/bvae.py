import tensorflow as tf
import tensorflow_probability as tfp
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from text_data_utils import *


def ret_recon_loss_oh_bow(true, pred_prob, vocab_size):
    true_bag_of_words = docs_to_oh_bags_of_words(true, vocab_size)
    recon_all = tf.nn.sigmoid_cross_entropy_with_logits(tf.stop_gradient(true_bag_of_words), pred_prob)
    recon = tf.reduce_mean(tf.reduce_sum(recon_all, axis=-1)) / 100
    return recon


def ret_recon_loss_bow(true, pred_prob, vocab_size):
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
    def __init__(self, language_dim, l_vocab_size, l_emb_dim, source_code_dim, c_vocab_size, c_emb_dim,
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
            self.recon_loss = ret_recon_loss_oh_bow
        elif architecture == 'mlp_no_embeddings':
            l_enc_model = VariationalEncoderNoEmbedding
            c_enc_model = VariationalEncoderNoEmbedding
            l_dec_model = Decoder
            c_dec_model = Decoder
            self.recon_loss = ret_recon_loss_bow
        else:
            raise Exception("Invalid architecture specification %s" % architecture)
        self.language_encoder = l_enc_model(language_dim, latent_dim, l_vocab_size, l_emb_dim,
                                            input_dropout=input_dropout, name='language_encoder')
        self.source_code_encoder = c_enc_model(source_code_dim, latent_dim, c_vocab_size, c_emb_dim,
                                               input_dropout=input_dropout, name='source_code_encoder')
        self.language_decoder = l_dec_model(latent_dim, language_dim, l_vocab_size,
                                            name='language_decoder')
        self.source_code_decoder = c_dec_model(latent_dim, source_code_dim, c_vocab_size,
                                               name='source_code_decoder')
        self.l_dim = language_dim
        self.c_dim = source_code_dim
        self.l_vocab_size = l_vocab_size
        self.c_vocab_size = c_vocab_size

    def compute_and_add_loss(self, language_batch, source_code_batch, enc_source_code_dists, enc_language_dists,
                             dec_language, dec_source_code,
                             al=0.35, bl=0.15, ac=0.35, bc=0.15):
        mean_dists = dists_means(enc_language_dists, enc_source_code_dists)
        language_kld = mpreg_loss(enc_language_dists, mean_dists)
        source_code_kld = mpreg_loss(enc_source_code_dists, mean_dists)
        """language_kld = preg_loss(enc_language_dists, enc_source_code_dists)
        source_code_kld = preg_loss(enc_source_code_dists, enc_language_dists)"""
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


def create_bvae(model_path):
    if not os.path.isfile(model_path + "model_description.json"):
        raise FileNotFoundError("Model description not found")

    with open(model_path + "model_description.json", 'r') as json_file:
        model_description = json.load(json_file)

    l_dim = model_description['l_dim']
    l_vocab_size = model_description['l_vocab_size']
    l_emb_dim = model_description['l_emb_dim']
    c_dim = model_description['c_dim']
    c_vocab_size = model_description['c_vocab_size']
    c_emb_dim = model_description['c_emb_dim']
    latent_dim = model_description['latent_dim']
    dropout_rate = model_description['dropout_rate']
    architecture = model_description['architecture']

    model = BimodalVariationalAutoEncoder(l_dim, l_vocab_size, l_emb_dim, c_dim, c_vocab_size,
                                          c_emb_dim, latent_dim, input_dropout=dropout_rate,
                                          architecture=architecture)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=False)

    if os.path.isfile(model_path + "checkpoint"):
        model.load_weights(model_path + "model_checkpoint.ckpt")

    return model


def load_or_create_seqifier(file_path, vocab_size, training_texts, tokenization):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as json_file:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_file.read())
    else:
        assert training_texts is not None
        assert tokenization is not None
        training_texts = tokenization(training_texts)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<unk>',
                                                          filters='', lower=False)
        tokenizer.fit_on_texts(training_texts)
        out_json = tokenizer.to_json()
        with open(file_path, 'w') as json_file:
            json_file.write(out_json)

    assert tokenizer.num_words == vocab_size
    return tokenizer


def process_dataset(summaries, codes, language_seqifier, code_seqifier, l_dim, c_dim,
                    oversize_sequence_behavior='leave_out'):
    assert len(summaries) == len(codes)
    summaries_tok = tokenize_texts(summaries)
    summaries_seq = language_seqifier.texts_to_sequences(summaries_tok)
    codes_tok = parse_codes(codes, c_dim)
    codes_seq = code_seqifier.texts_to_sequences(codes_tok)
    summaries_trim, codes_trim = trim_to_len(summaries_seq, codes_seq, l_dim, c_dim,
                                             oversize_sequence_behavior=oversize_sequence_behavior)
    return summaries_trim, codes_trim


def train_bvae(model, model_path, train_summaries, train_codes, val_summaries, val_codes):

    checkpoints = tf.keras.callbacks.ModelCheckpoint(model_path + 'model_checkpoint.ckpt',
                                                     verbose=True, save_best_only=True,
                                                     monitor='val_loss', save_freq='epoch', save_weights_only=True)
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit((train_summaries, train_codes), None, batch_size=128, epochs=6,
                        validation_data=((val_summaries, val_codes), None),
                        callbacks=[checkpoints, reduce_on_plateau, early_stopping])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(model_path + 'performance_plot.png')


class RetBVAE(object):
    def __init__(self, model, code_snippets, language_seqifier, code_seqifier):
        self.model = model
        self.raw_codes = code_snippets
        codes_tok = parse_codes(code_snippets, model.c_dim)
        codes_seq = code_seqifier.texts_to_sequences(codes_tok)
        codes_padded = pad_sequences(codes_seq, maxlen=model.c_dim, padding='post', value=0)
        self.codes = model.source_code_encoder(codes_padded)
        self.code_snippets = code_snippets
        self.language_seqifier = language_seqifier
        self.code_seqifier = code_seqifier

    def get_similarities(self, query):
        query_prep = preprocess_language(query)
        query_tok = tokenize_text(query_prep)
        query_seq = self.language_seqifier.texts_to_sequences([query_tok])
        query_padded = pad_sequences(query_seq, maxlen=self.model.l_dim, padding='post', value=0)
        query_encoded = self.model.language_encoder(query_padded)
        similarities_all = tfp.distributions.kl_divergence(
            query_encoded,
            self.codes
        )
        similarities = tf.reduce_sum(similarities_all, axis=-1)
        return similarities

    def rank_options(self, query):
        similarities = self.get_similarities(query)
        ranked_indices = tf.argsort(similarities, direction='ASCENDING').numpy()
        return ranked_indices

    def interactive_demo(self):
        while True:
            input_summary = input("Input Summary: ")
            if input_summary == "exit":
                break
            ranked_options = self.rank_options(input_summary)
            print("Retrieved Code: %s" % self.raw_codes[ranked_options[0]])


def main(model_path='../models/r5/'):

    train_summaries, train_codes = load_iyer_file("../data/iyer_csharp/train.txt")
    val_summaries, val_codes = load_iyer_file("../data/iyer_csharp/valid.txt")
    test_summaries, test_codes = load_iyer_file("../data/iyer_csharp/test.txt")

    model = create_bvae(model_path)
    language_tokenizer = load_or_create_seqifier(model_path + "language_tokenizer.json",
                                                 model.l_vocab_size, train_summaries,
                                                 lambda s: tokenize_texts(s))
    code_tokenizer = load_or_create_seqifier(model_path + "code_tokenizer.json",
                                             model.c_vocab_size, train_codes,
                                             lambda c: parse_codes(c, model.c_dim))

    train_summaries, train_codes = process_dataset(train_summaries, train_codes, language_tokenizer, code_tokenizer,
                                                   model.l_dim, model.c_dim)
    val_summaries, val_codes = process_dataset(val_summaries, val_codes, language_tokenizer, code_tokenizer,
                                               model.l_dim, model.c_dim)
    test_summaries, test_codes = process_dataset(test_summaries, test_codes, language_tokenizer, code_tokenizer,
                                                 model.l_dim, model.c_dim)

    if not os.path.isfile(model_path + "checkpoint"):
        train_bvae(model, model_path, train_summaries, train_codes, val_summaries, val_codes)

    test_loss = model.evaluate((test_summaries, test_codes), None, verbose=False)
    print("Test loss: " + str(test_loss))

    dev_summaries, dev_codes = load_iyer_file("../data/iyer_csharp/dev.txt")

    ret_bvae = RetBVAE(model, dev_codes, language_tokenizer, code_tokenizer)
    ret_bvae.interactive_demo()


if __name__ == "__main__":
    main()
