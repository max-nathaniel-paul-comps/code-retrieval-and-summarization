import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import os
import json
import text_data_utils as tdu
from tokenizer import Tokenizer
import transformer


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def recon_loss_bow(true, pred_prob, vocab_size):
    mask = tf.logical_not(tf.equal(true, 0))
    true_ragged = tf.ragged.boolean_mask(true, mask)
    true_one_hot = tf.one_hot(true_ragged, vocab_size, axis=-1)
    true_bags_of_words = tf.reduce_mean(true_one_hot, axis=-2)
    recon_all = tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(true_bags_of_words), pred_prob, axis=-1)
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


class Dropout(tf.keras.layers.Layer):
    def __init__(self, rate, name='dropout'):
        super(Dropout, self).__init__(name=name)
        self.rate = rate

    def call(self, inputs, training=False, **kwargs):
        if training:
            noise = tf.random.uniform(tf.shape(inputs), minval=0, maxval=1, dtype=tf.float32)
            noise_ints = tf.cast(tf.greater_equal(noise, self.rate), tf.int32)
            outputs = inputs * noise_ints
            return outputs
        else:
            return inputs


class Raggify(tf.keras.layers.Layer):
    def __init__(self, name='raggify'):
        super(Raggify, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        mask = tf.logical_not(tf.equal(inputs, 0))
        inputs_ragged = tf.ragged.boolean_mask(inputs, mask)
        return inputs_ragged


class MlpBowEncoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, vocab_size, emb_dim, name='variational_encoder', **kwargs):
        super(MlpBowEncoder, self).__init__(
            [
                Raggify(),
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


class RecurrentEncoder(tf.keras.Model):
    def __init__(self, latent_dim, vocab_size, emb_dim, input_dropout_rate=0.0, name='gru_variational_encoder',
                 **kwargs):
        super(RecurrentEncoder, self).__init__(name=name)
        self.dropout = Dropout(input_dropout_rate)
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.gru = tf.keras.layers.GRU(latent_dim * 2, return_sequences=False, return_state=True)
        self.projection = tf.keras.layers.Dense(latent_dim * 2)
        self.latent_dim = latent_dim

    def call(self, inputs, training=False, **kwargs):
        mask = tf.logical_not(tf.equal(inputs, 0))
        dropped = self.dropout(inputs, training=training)
        embedded = self.embedding(dropped)
        _, final_gru_state = self.gru(embedded, mask=mask)
        projected = self.projection(final_gru_state)
        dists = create_latent_dists(projected, self.latent_dim)
        return dists


class TransformerEncoder(tf.keras.Model):
    def __init__(self, latent_dim, vocab_size, emb_dim, name='variational_encoder', **kwargs):
        super(TransformerEncoder, self).__init__(name=name)
        self.transformer_encoder = transformer.UTEncoder(4, 128, 8, 512, vocab_size, 1024)
        self.down_projection = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim * 2)
        ])
        self.latent_dim = latent_dim

    def call(self, inputs, training=False, **kwargs):
        mask = transformer.create_padding_mask(inputs)
        transformed = self.transformer_encoder(inputs, training, mask)
        projected = self.down_projection(transformed)
        dists = create_latent_dists(projected, self.latent_dim)
        return dists


class MlpBowDecoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, vocab_size, emb_dim, teacher_dropout_rate, start_token, end_token,
                 name='decoder', **kwargs):
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
                 name='gru_decoder', **kwargs):
        super(RecurrentDecoder, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.teacher_dropout = Dropout(teacher_dropout_rate)
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.dense = tf.keras.layers.Dense(latent_dim * 2)
        self.gru = tf.keras.layers.GRU(latent_dim * 2, return_sequences=True)
        self.dense_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size))

    def teacher_forcing_decode(self, latent_samples, true_outputs, training=False):
        teacher_slice = true_outputs[:, :-1]
        teacher_mask = tf.logical_not(tf.equal(teacher_slice, 0))
        teacher_dropped = self.teacher_dropout(teacher_slice, training=training)
        teacher_embedded = self.embedding(teacher_dropped)
        dense_out = self.dense(latent_samples)
        gru_out = self.gru(teacher_embedded, initial_state=dense_out, training=training, mask=teacher_mask)
        predicts = self.dense_2(gru_out)
        return predicts

    def beam_search_decode(self, latent_samples, beam_width=10, max_len=50):
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


class TransformerDecoder(tf.keras.Model):
    def __init__(self, latent_dim, vocab_size, emb_dim, teacher_dropout_rate, start_token, end_token,
                 reconstructed_dim=0, name='decoder', **kwargs):
        super(TransformerDecoder, self).__init__(name=name)
        self.up_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(reconstructed_dim * 32),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((reconstructed_dim, 32)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128))
        ])
        self.teacher_dropout = Dropout(teacher_dropout_rate)
        self.transformer_decoder = transformer.UTDecoder(4, 128, 8, 512, vocab_size, reconstructed_dim)
        self.look_ahead_mask = transformer.create_look_ahead_mask(reconstructed_dim)
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
        self.start_token = start_token
        self.end_token = end_token
        self.reconstructed_dim = reconstructed_dim

    def teacher_forcing_decode(self, latent_samples, true_outputs, training):
        teacher_slice = true_outputs[:, :-1]
        teacher_dropped = self.teacher_dropout(teacher_slice, training=training)
        projected = self.up_projection(latent_samples)
        padding_mask = transformer.create_padding_mask(teacher_dropped)
        transformed, _ = self.transformer_decoder(teacher_dropped, projected, training, self.look_ahead_mask,
                                                  padding_mask)
        outputs = self.dense(transformed)
        return outputs

    def beam_search_decode(self, latent_samples):
        projected = self.up_projection(latent_samples)
        outputs = []
        for example in projected:
            output = [self.start_token]
            for i in range(self.reconstructed_dim):
                look_ahead_mask = tf.zeros((1, 1), dtype=tf.float32)
                output_tensor = tf.expand_dims(output, 0)
                padding_mask = tf.zeros((1, 1, 1, 1), dtype=tf.float32)
                transformed, _ = self.transformer_decoder(output_tensor, tf.expand_dims(example, 0), False,
                                                          look_ahead_mask, padding_mask)
                prediction = tf.argmax(self.dense(transformed), axis=-1)[0][-1].numpy()
                output.append(prediction)
                if output[-1] == self.end_token:
                    break
            outputs.append(output)
        return outputs

    def call(self, latent_samples, true_outputs=None, training=False, **kwargs):
        if true_outputs is not None:
            return self.teacher_forcing_decode(latent_samples, true_outputs, training)
        else:
            return self.beam_search_decode(latent_samples)


encoders = {
    'mlp_bow': MlpBowEncoder,
    'recurrent': RecurrentEncoder,
    'transformer': TransformerEncoder
}

decoders = {
    'mlp_bow': MlpBowDecoder,
    'recurrent': RecurrentDecoder,
    'transformer': TransformerDecoder
}

optimizers = {
    'adam': tf.keras.optimizers.Adam(learning_rate=0.001)
}

recon_losses = {
    'bow': recon_loss_bow,
    'full': recon_loss
}


class BimodalVariationalAutoEncoder(tf.Module):
    def __init__(self, model_path, tokenizers_training_texts=(None, None), tf_name='bvae'):

        super(BimodalVariationalAutoEncoder, self).__init__(name=tf_name)

        if not os.path.isfile(model_path + "model_description.json"):
            raise FileNotFoundError("Model description not found")

        with open(model_path + "model_description.json", 'r') as json_file:
            model_description = json.load(json_file)

        self.language_tokenizer = Tokenizer(model_description['language_tokenizer_type'],
                                            model_path + model_description['language_tokenizer_path'],
                                            training_texts=tokenizers_training_texts[0],
                                            target_vocab_size=model_description['language_target_vocab_size'],
                                            vocab_min_count=model_description['language_vocab_min_count'])
        self.code_tokenizer = Tokenizer(model_description['code_tokenizer_type'],
                                        model_path + model_description['code_tokenizer_path'],
                                        training_texts=tokenizers_training_texts[1],
                                        target_vocab_size=model_description['code_target_vocab_size'],
                                        vocab_min_count=model_description['code_vocab_min_count'])

        self.l_dim = model_description['l_dim']
        self.c_dim = model_description['c_dim']

        self.latent_dim = model_description['latent_dim']
        self.kld_loss_type = model_description['kld_loss_type']

        self.language_encoder = encoders[model_description['l_enc_type']](
            self.latent_dim,
            self.language_tokenizer.vocab_size,
            model_description['l_emb_dim'],
            input_dropout_rate=model_description['l_enc_dropout'],
            name='language_encoder'
        )

        self.source_code_encoder = encoders[model_description['c_enc_type']](
            self.latent_dim,
            self.code_tokenizer.vocab_size,
            model_description['c_emb_dim'],
            input_dropout_rate=model_description['c_enc_dropout'],
            name='source_code_encoder'
        )

        self.language_decoder = decoders[model_description['l_dec_type']](
            self.latent_dim,
            self.language_tokenizer.vocab_size,
            model_description['l_emb_dim'],
            model_description['l_dec_dropout'],
            self.language_tokenizer.start_token,
            self.language_tokenizer.end_token,
            reconstructed_dim=self.l_dim - 1,
            name='language_decoder'
        )

        self.source_code_decoder = decoders[model_description['c_dec_type']](
            self.latent_dim,
            self.code_tokenizer.vocab_size,
            model_description['c_emb_dim'],
            model_description['c_dec_dropout'],
            self.code_tokenizer.start_token,
            self.code_tokenizer.end_token,
            reconstructed_dim=self.c_dim - 1,
            name='source_code_decoder'
        )

        self.language_recon_loss = recon_losses[model_description['l_recon_loss']]
        self.source_code_recon_loss = recon_losses[model_description['c_recon_loss']]

        self.optimizer = optimizers[model_description['optimizer']]

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
        language_recon = self.language_recon_loss(summaries, dec_language, self.language_tokenizer.vocab_size)
        source_code_recon = self.source_code_recon_loss(codes, dec_source_code, self.code_tokenizer.vocab_size)
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

    def train(self, train_summaries, train_codes, val_summaries, val_codes, num_epochs=100, batch_size=64, patience=6):

        assert len(train_summaries) == len(train_codes)

        print("Tokenizing datasets, and removing examples that are too long...")
        train_summaries = self.language_tokenizer.tokenize_texts(train_summaries)
        train_codes = self.code_tokenizer.tokenize_texts(train_codes)
        train_summaries, train_codes = tdu.sequences_to_tensors(train_summaries, train_codes, self.l_dim, self.c_dim)
        val_summaries = self.language_tokenizer.tokenize_texts(val_summaries)
        val_codes = self.code_tokenizer.tokenize_texts(val_codes)
        val_summaries, val_codes = tdu.sequences_to_tensors(val_summaries, val_codes, self.l_dim, self.c_dim)

        dataset = tf.data.Dataset.from_tensor_slices((train_summaries, train_codes))

        len_train = train_summaries.shape[0]
        batches_per_epoch = int(len_train / batch_size)
        print("Training on %s samples, validating on %s samples" % (len_train, val_summaries.shape[0]))

        best_val_loss = self.evaluate(val_summaries, val_codes, batch_size=batch_size)
        print("Initial validation loss: %s" % best_val_loss.numpy())
        num_epochs_with_no_improvement = 0

        for epoch in range(num_epochs):

            shuffled = dataset.shuffle(len_train, reshuffle_each_iteration=True)
            shuffled_batches = shuffled.batch(batch_size, drop_remainder=True)
            batches_iter = iter(shuffled_batches)

            batches = tqdm.trange(batches_per_epoch)
            for batch in batches:
                summaries_batch, codes_batch = next(batches_iter)
                loss = self.training_step(summaries_batch,
                                          codes_batch,
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

    def summaries_to_latent(self, summaries):
        tokenized = self.language_tokenizer.tokenize_texts(summaries)
        if len(tokenized[0]) > self.l_dim:
            print("Warning: Input summary is oversize")
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=self.l_dim, padding='post', value=0)
        latent = self.language_encoder(padded, training=False)
        return latent

    def codes_to_latent(self, codes):
        tokenized = self.code_tokenizer.tokenize_texts(codes)
        if len(tokenized[0]) > self.c_dim:
            print("Warning: Input code is oversize")
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=self.l_dim, padding='post', value=0)
        latent = self.source_code_encoder(padded, training=False)
        return latent

    def latent_to_summaries(self, latent):
        mean = latent.mean()
        tokenized = self.language_decoder(mean, training=False)
        summaries = self.language_tokenizer.de_tokenize_texts(tokenized)
        return summaries

    def latent_to_codes(self, latent):
        mean = latent.mean()
        tokenized = self.source_code_decoder(mean, training=False)
        codes = self.code_tokenizer.de_tokenize_texts(tokenized)
        return codes
