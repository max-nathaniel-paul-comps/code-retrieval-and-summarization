import tensorflow as tf
import tensorflow_probability as tfp
import os
import json


def recon_loss_bow(true, pred_prob, vocab_size):
    true_one_hot = tf.one_hot(true, vocab_size)
    true_bags_of_words = tf.reduce_mean(true_one_hot, axis=-2)
    recon_all = tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(true_bags_of_words), pred_prob)
    recon = tf.reduce_mean(recon_all)
    return recon


def recon_loss(true, pred, _):
    true_slice = true[:, 1:]
    mask = tf.logical_not(tf.equal(true_slice, 0))
    recon_all = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.stop_gradient(true_slice), pred)
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


class RecurrentEncoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, vocab_size, name='gru_variational_encoder'):
        super(RecurrentEncoder, self).__init__(name=name)
        self.gru = tf.keras.layers.GRU(latent_dim, return_sequences=False, return_state=True, go_backwards=True)
        self.dense = tf.keras.layers.Dense(latent_dim * 2)
        self.latent_dim = latent_dim

    def call(self, x, training=False, **kwargs):
        gru_out, gru_state = self.gru(x)
        latent_raw = self.dense(gru_state)
        latent = create_latent_dists(latent_raw, self.latent_dim)
        return latent


class MlpBowEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, vocab_size, name='variational_encoder'):
        super(MlpBowEncoder, self).__init__(
            [
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Activation('tanh'),
                tf.keras.layers.Dense(latent_dim * 2, name='emb_to_hidden'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(latent_dim * 2, name='hidden_to_encoded')
            ],
            name=name
        )
        self.latent_dim = latent_dim

    def call(self, x, training=None, **kwargs):
        raw_dists = super(MlpBowEncoder, self).call(x, training=training)
        dists = create_latent_dists(raw_dists, self.latent_dim)
        return dists


class MlpBowDecoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, reconstructed_dim, vocab_size, embedding, start_token, end_token,
                 name='decoder'):
        super(MlpBowDecoder, self).__init__(
            [
                tf.keras.layers.Dense(latent_dim * 2, input_dim=latent_dim, name='encoded_to_hidden'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(vocab_size, name='hidden_to_decoded')
            ],
            name=name
        )

    def call(self, inputs, training=None, **kwargs):
        return super(MlpBowDecoder, self).call(inputs, training=training)


class RecurrentDecoder(tf.keras.Model):
    def __init__(self, latent_dim, reconstructed_dim, vocab_size, embedding, start_token, end_token,
                 name='gru_decoder'):
        super(RecurrentDecoder, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.reconstructed_dim = reconstructed_dim - 1
        self.embedding = embedding
        self.start_token = start_token
        self.end_token = end_token
        self.gru = tf.keras.layers.GRU(latent_dim, return_sequences=True)
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size))

    def teacher_forcing_decode(self, latent_samples, true_outputs):
        teacher_slice = true_outputs[:, :true_outputs.shape[1] - 1, :]
        gru_out = self.gru(teacher_slice, initial_state=latent_samples)
        predicts = self.dense(gru_out)
        return predicts

    def beam_search_decode(self, latent_samples):  # TODO actually implement beam search
        predicted_texts = []
        for i in range(latent_samples.shape[0]):
            predicted_token = self.start_token
            predicted_text = [predicted_token]
            state = [tf.expand_dims(tf.expand_dims(latent_samples[i], 0), 0)]
            for j in range(self.reconstructed_dim):
                predicted_embedded = self.embedding(tf.expand_dims(tf.expand_dims(predicted_token, 0), 0))
                output, state = self.gru.cell(predicted_embedded, state)
                predicted_token = tf.argmax(self.dense(output), axis=-1, output_type=tf.int32).numpy()[0][0]
                predicted_text += [predicted_token]
                if predicted_token == self.end_token:
                    break
            predicted_texts += [predicted_text]
        return predicted_texts

    def call(self, latent_samples, true_outputs=None, **kwargs):
        if true_outputs is not None:
            return self.teacher_forcing_decode(latent_samples, true_outputs)
        else:
            return self.beam_search_decode(latent_samples)


encoders = {
    'mlp_bow': MlpBowEncoder,
    'recurrent': RecurrentEncoder
}

decoders = {
    'mlp_bow': MlpBowDecoder,
    'recurrent': RecurrentDecoder
}

recon_losses = {
    'bow': recon_loss_bow,
    'full': recon_loss
}


class BimodalVariationalAutoEncoder(tf.keras.Model):
    def __init__(self, model_path, language_seqifier, code_seqifier, tf_name='bvae'):
        super(BimodalVariationalAutoEncoder, self).__init__(name=tf_name)

        if not os.path.isfile(model_path + "model_description.json"):
            raise FileNotFoundError("Model description not found")

        with open(model_path + "model_description.json", 'r') as json_file:
            model_description = json.load(json_file)

        self.l_dim = model_description['l_dim']
        self.c_dim = model_description['c_dim']
        self.l_vocab_size = language_seqifier.vocab_size
        self.c_vocab_size = code_seqifier.vocab_size
        self.latent_dim = model_description['latent_dim']
        self.kld_loss_type = model_description['kld_loss_type']

        self.language_embedding = tf.keras.layers.Embedding(self.l_vocab_size, model_description['l_emb_dim'])
        self.source_code_embedding = tf.keras.layers.Embedding(self.c_vocab_size, model_description['c_emb_dim'])

        self.language_dropout = tf.keras.layers.Dropout(model_description['input_dropout'])
        self.source_code_dropout = tf.keras.layers.Dropout(model_description['input_dropout'])

        self.language_encoder = encoders[model_description['l_enc_type']](
            self.l_dim,
            self.latent_dim,
            self.l_vocab_size,
            name='language_encoder'
        )

        self.source_code_encoder = encoders[model_description['c_enc_type']](
            self.c_dim,
            self.latent_dim,
            self.c_vocab_size,
            name='source_code_encoder'
        )

        self.language_decoder = decoders[model_description['l_dec_type']](
            self.latent_dim,
            self.l_dim,
            self.l_vocab_size,
            self.language_embedding,
            language_seqifier.start_token,
            language_seqifier.end_token,
            name='language_decoder'
        )

        self.source_code_decoder = decoders[model_description['c_dec_type']](
            self.latent_dim,
            self.c_dim,
            self.c_vocab_size,
            self.source_code_embedding,
            code_seqifier.start_token,
            code_seqifier.end_token,
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

        language_batch_emb = self.language_embedding(language_batch)
        source_code_batch_emb = self.source_code_embedding(source_code_batch)
        language_batch_emb_drop = self.language_dropout(language_batch_emb, training=training)
        source_code_batch_emb_drop = self.source_code_dropout(source_code_batch_emb, training=training)

        enc_language_dists = self.language_encoder(language_batch_emb_drop, training=training)
        enc_source_code_dists = self.source_code_encoder(source_code_batch_emb_drop, training=training)
        enc_language = enc_language_dists.sample()
        enc_source_code = enc_source_code_dists.sample()

        dec_language = self.language_decoder(enc_language, training=training,
                                             true_outputs=language_batch_emb_drop)
        dec_source_code = self.source_code_decoder(enc_source_code, training=training,
                                                   true_outputs=source_code_batch_emb_drop)

        self.compute_and_add_loss(language_batch, source_code_batch, enc_source_code_dists, enc_language_dists,
                                  dec_language, dec_source_code)

        return dec_language, dec_source_code
