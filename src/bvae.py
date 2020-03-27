import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import os
import json
import numpy as np


GLOBAL_MAX_SEQUENCE_LEN = 1200


def recon_loss_bow(true, pred_prob, vocab_size):
    true_one_hot = tf.one_hot(true, vocab_size, axis=-1)
    true_bags_of_words = tf.reduce_mean(true_one_hot, axis=-2)
    recon_all = tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(true_bags_of_words), pred_prob, axis=-1)
    recon = tf.reduce_mean(recon_all)
    return recon


def recon_loss(true, pred, _):
    true_slice = true[:, 1:]
    true_slice_non_ragged = true_slice.to_tensor()
    pred_non_ragged = pred.to_tensor()
    mask = tf.logical_not(tf.equal(true_slice_non_ragged, 0))
    recon_all = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.stop_gradient(true_slice_non_ragged), pred_non_ragged)
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


def get_angles(pos, i, d_model):  # Lifted from TensorFlow Transformer tutorial
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):  # Also lifted from TensorFlow Transformer tutorial
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, dim, name='pos_enc'):
        super(PositionalEncoding, self).__init__(name=name)
        self._supports_ragged_inputs = True
        self.encodings = positional_encoding(GLOBAL_MAX_SEQUENCE_LEN, dim)

    def call(self, inputs, **kwargs):
        inputs_non_ragged = inputs.to_tensor()
        mask = tf.logical_not(tf.reduce_all(tf.equal(inputs_non_ragged, 0), axis=-1))
        result = inputs_non_ragged + self.encodings[:, :tf.shape(inputs_non_ragged)[1], :]
        result_ragged = tf.ragged.boolean_mask(result, mask)
        return result_ragged


class RaggedDropout(tf.keras.layers.Layer):
    def __init__(self, rate, name='ragged_dropout'):
        super(RaggedDropout, self).__init__(name=name)
        self._supports_ragged_inputs = True
        self.rate = rate

    def call(self, inputs, training=False, **kwargs):
        if training:
            noise = tf.random.uniform(tf.shape(inputs.values), minval=0, maxval=1, dtype=tf.float32)
            noise_ints = tf.cast(tf.greater_equal(noise, self.rate), tf.int32)
            output_values = inputs.values * noise_ints
            outputs = tf.RaggedTensor.from_row_splits(output_values, inputs.row_splits)
            return outputs
        else:
            return inputs


class RecurrentEncoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, vocab_size, emb_dim, input_dropout_rate, name='gru_variational_encoder'):
        super(RecurrentEncoder, self).__init__(
            [
                tf.keras.layers.Input(shape=(None,), ragged=True, dtype=tf.int32),
                RaggedDropout(input_dropout_rate),
                tf.keras.layers.Embedding(vocab_size, emb_dim),
            ],
            name=name
        )
        self.gru = tf.keras.layers.GRU(latent_dim * 2, return_sequences=False, return_state=True, go_backwards=True)
        self.gru.could_use_cudnn = False
        self.dense = tf.keras.layers.Dense(latent_dim * 2)
        self.latent_dim = latent_dim

    def call(self, x, training=False, **kwargs):
        embedded_and_dropped = super(RecurrentEncoder, self).call(x, training=training)
        gru_out, gru_state = self.gru(embedded_and_dropped)
        latent_raw = self.dense(gru_state)
        latent = create_latent_dists(latent_raw, self.latent_dim)
        return latent


class MlpBowEncoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, vocab_size, emb_dim, input_dropout_rate, name='variational_encoder'):
        super(MlpBowEncoder, self).__init__(
            [
                tf.keras.layers.Input(shape=(None,), ragged=True, dtype=tf.int32),
                RaggedDropout(input_dropout_rate),
                tf.keras.layers.Embedding(vocab_size, emb_dim),
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


class MlpPeEncoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, vocab_size, emb_dim, input_dropout_rate, name='variational_encoder'):
        super(MlpPeEncoder, self).__init__(
            [
                tf.keras.layers.Input(shape=(None,), ragged=True, dtype=tf.int32),
                RaggedDropout(input_dropout_rate),
                tf.keras.layers.Embedding(vocab_size, emb_dim),
                PositionalEncoding(emb_dim),
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
        raw_dists = super(MlpPeEncoder, self).call(x, training=training)
        dists = create_latent_dists(raw_dists, self.latent_dim)
        return dists


class MlpBowDecoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, vocab_size, emb_dim, teacher_dropout_rate, start_token, end_token,
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
    def __init__(self, latent_dim, vocab_size, emb_dim, teacher_dropout_rate, start_token, end_token,
                 name='gru_decoder'):
        super(RecurrentDecoder, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.teacher_dropout = RaggedDropout(teacher_dropout_rate)
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.dense = tf.keras.layers.Dense(latent_dim * 2)
        self.gru = tf.keras.layers.GRU(latent_dim * 2, return_sequences=True)
        self.gru.could_use_cudnn = False
        self.dense_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size))

    def teacher_forcing_decode(self, latent_samples, true_outputs, training=False):
        teacher_slice = true_outputs[:, :-1]
        teacher_dropped = self.teacher_dropout(teacher_slice, training=training)
        teacher_embedded = self.embedding(teacher_dropped)
        dense_out = self.dense(latent_samples)
        gru_out = self.gru(teacher_embedded, initial_state=dense_out, training=training)
        predicts = self.dense_2(gru_out)
        return predicts

    def beam_search_decode(self, latent_samples, beam_width=10, max_len=GLOBAL_MAX_SEQUENCE_LEN):
        predicted_texts = []
        for i in range(latent_samples.shape[0]):
            dense_out = self.dense(tf.expand_dims(latent_samples[i], 0))
            predicted_embedded = self.embedding(tf.expand_dims(tf.expand_dims(self.start_token, 0), 0))
            output, initial_state = self.gru.cell(predicted_embedded, tf.expand_dims(dense_out, 0))
            final = tf.nn.softmax(self.dense_2(output)[0][0], axis=-1)
            predictions = tf.argsort(final, axis=-1, direction='DESCENDING').numpy()[0:beam_width]
            beams = []
            for k in range(beam_width):
                formed_candidate = ([self.start_token, predictions[k]],
                                    -tf.math.log(final[predictions[k]]),
                                    initial_state)
                beams.append(formed_candidate)
            for j in range(max_len - 1):
                candidates = []
                for k in range(beam_width):
                    if beams[k][0][-1] == self.end_token:
                        if len(candidates) < beam_width:
                            candidates.append(beams[k])
                        else:
                            for m in range(len(candidates)):
                                if candidates[m][1] > beams[k][1]:
                                    candidates[m] = beams[k]
                                    break
                    else:
                        predicted_embedded = self.embedding(tf.expand_dims(tf.expand_dims(beams[k][0][-1], 0), 0))
                        output, new_state = self.gru.cell(predicted_embedded, beams[k][2])
                        final = tf.nn.softmax(self.dense_2(output)[0][0], axis=-1)
                        predictions = tf.argsort(final, axis=-1, direction='DESCENDING').numpy()[0:beam_width]
                        for prediction in predictions:
                            formed_candidate = (beams[k][0] + [prediction],
                                                beams[k][1] + -tf.math.log(final[prediction]),
                                                new_state)
                            if len(candidates) < beam_width:
                                candidates.append(formed_candidate)
                            else:
                                for m in range(len(candidates)):
                                    if candidates[m][1] > formed_candidate[1]:
                                        candidates[m] = formed_candidate
                                        break
                beams = candidates
                if all(beams[k][0][-1] == self.end_token for k in range(beam_width)):
                    break
            lowest_perplexity_beam = beams[0]
            for k in range(1, beam_width):
                if beams[k][1] < lowest_perplexity_beam[1]:
                    lowest_perplexity_beam = beams[k]
            predicted_texts.append(lowest_perplexity_beam[0])
        return predicted_texts

    def call(self, latent_samples, true_outputs=None, training=False, **kwargs):
        if true_outputs is not None:
            return self.teacher_forcing_decode(latent_samples, true_outputs, training=training)
        else:
            return self.beam_search_decode(latent_samples)


encoders = {
    'mlp_bow': MlpBowEncoder,
    'recurrent': RecurrentEncoder,
    'mlp_pe': MlpPeEncoder
}

decoders = {
    'mlp_bow': MlpBowDecoder,
    'recurrent': RecurrentDecoder
}

recon_losses = {
    'bow': recon_loss_bow,
    'full': recon_loss
}


class BimodalVariationalAutoEncoder(tf.Module):
    def __init__(self, model_path, language_seqifier, code_seqifier,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), tf_name='bvae'):

        super(BimodalVariationalAutoEncoder, self).__init__(name=tf_name)

        if not os.path.isfile(model_path + "model_description.json"):
            raise FileNotFoundError("Model description not found")

        with open(model_path + "model_description.json", 'r') as json_file:
            model_description = json.load(json_file)

        self.l_vocab_size = language_seqifier.vocab_size
        self.c_vocab_size = code_seqifier.vocab_size
        self.latent_dim = model_description['latent_dim']
        self.kld_loss_type = model_description['kld_loss_type']

        self.language_encoder = encoders[model_description['l_enc_type']](
            self.latent_dim,
            self.l_vocab_size,
            model_description['l_emb_dim'],
            model_description['l_enc_dropout'],
            name='language_encoder'
        )

        self.source_code_encoder = encoders[model_description['c_enc_type']](
            self.latent_dim,
            self.c_vocab_size,
            model_description['c_emb_dim'],
            model_description['c_enc_dropout'],
            name='source_code_encoder'
        )

        self.language_decoder = decoders[model_description['l_dec_type']](
            self.latent_dim,
            self.l_vocab_size,
            model_description['l_emb_dim'],
            model_description['l_dec_dropout'],
            language_seqifier.start_token,
            language_seqifier.end_token,
            name='language_decoder'
        )

        self.source_code_decoder = decoders[model_description['c_dec_type']](
            self.latent_dim,
            self.c_vocab_size,
            model_description['c_emb_dim'],
            model_description['c_dec_dropout'],
            code_seqifier.start_token,
            code_seqifier.end_token,
            name='source_code_decoder'
        )

        self.language_recon_loss = recon_losses[model_description['l_recon_loss']]
        self.source_code_recon_loss = recon_losses[model_description['c_recon_loss']]

        self.optimizer = optimizer
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_path + "checkpoints/", max_to_keep=3)
        checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def loss(self, summaries, codes, al=0.35, bl=0.15, ac=0.35, bc=0.15):
        enc_language_dists = self.language_encoder(summaries, training=True)
        enc_source_code_dists = self.source_code_encoder(codes, training=True)
        enc_language = enc_language_dists.sample()
        enc_source_code = enc_source_code_dists.sample()
        dec_language = self.language_decoder(enc_language, training=True,
                                             true_outputs=summaries)
        dec_source_code = self.source_code_decoder(enc_source_code, training=True,
                                                   true_outputs=codes)
        if self.kld_loss_type == 'preg':
            language_kld = preg_loss(enc_language_dists, enc_source_code_dists)
            source_code_kld = preg_loss(enc_source_code_dists, enc_language_dists)
        elif self.kld_loss_type == 'mpreg':
            mean_dists = dists_means(enc_language_dists, enc_source_code_dists)
            language_kld = mpreg_loss(enc_language_dists, mean_dists)
            source_code_kld = mpreg_loss(enc_source_code_dists, mean_dists)
        else:
            raise Exception("Invalid KL-divergence loss: %s" % self.kld_loss_type)
        language_recon = self.language_recon_loss(summaries, dec_language, self.l_vocab_size)
        source_code_recon = self.source_code_recon_loss(codes, dec_source_code, self.c_vocab_size)
        final_loss = al * language_recon + bl * language_kld + ac * source_code_recon + bc * source_code_kld
        return final_loss

    @tf.function
    def training_step(self, summaries, codes, optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss(summaries,
                             codes)
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    @tf.function
    def evaluate(self, summaries, codes, batch_size=128):
        assert summaries.shape[0] == codes.shape[0]
        len_val = summaries.shape[0]
        batches_per_val = tf.cast(len_val / batch_size, tf.int32)
        val_loss = tf.convert_to_tensor(0.0, dtype=tf.float32)
        for val_batch in tf.range(0, batches_per_val, dtype=tf.int32):
            val_loss += self.loss(summaries[val_batch * batch_size: val_batch * batch_size + batch_size],
                                  codes[val_batch * batch_size: val_batch * batch_size + batch_size])
        val_loss /= tf.cast(batches_per_val, tf.float32)
        return val_loss

    def train(self, train_summaries, train_codes, val_summaries, val_codes, num_epochs=100, batch_size=128, patience=6):

        assert train_summaries.shape[0] == train_codes.shape[0]
        len_train = train_summaries.shape[0]
        batches_per_epoch = int(len_train / batch_size)
        print("Training on %s samples, validating on %s samples" % (len_train, val_summaries.shape[0]))

        best_val_loss = self.evaluate(val_summaries, val_codes, batch_size=batch_size)
        print("Initial validation loss: %s" % best_val_loss.numpy())
        num_epochs_with_no_improvement = 0

        for epoch in range(num_epochs):

            batches = tqdm.trange(batches_per_epoch)
            for batch in batches:
                loss = self.training_step(train_summaries[batch * batch_size: batch * batch_size + batch_size],
                                          train_codes[batch * batch_size: batch * batch_size + batch_size],
                                          self.optimizer)
                batches.set_description("Epoch %s of %s, loss=%.4f" % (epoch + 1, num_epochs, loss.numpy()))

            val_loss = self.evaluate(val_summaries, val_codes, batch_size=batch_size)
            print("Epoch %s of %s, val_loss=%s" % (epoch + 1, num_epochs, val_loss.numpy()))
            if val_loss < best_val_loss:
                print("Val loss improved from %s, saving checkpoint" % best_val_loss.numpy())
                self.checkpoint_manager.save()
                best_val_loss = val_loss
                num_epochs_with_no_improvement = 0
            else:
                print("Val loss did not improve")
                num_epochs_with_no_improvement += 1
                if num_epochs_with_no_improvement > patience:
                    print("I ran out of patience")
                    break
                if num_epochs_with_no_improvement % 2 == 0:
                    print("Decreasing learning rate by 80 percent")
                    self.optimizer.learning_rate.assign(self.optimizer.learning_rate * 0.2)
