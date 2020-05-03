import tensorflow as tf
import tensorflow_probability as tfp
import os
import json
import tqdm
from tokenizer import Tokenizer
import transformer
from tf_utils import dataset_to_batched_tensors, beam_search_decode, beam_search_decode_new
import text_data_utils as tdu


def recon_loss_bow(true, pred_prob, vocab_size):
    mask = tf.logical_not(tf.equal(true, 0))
    true_ragged = tf.ragged.boolean_mask(true, mask)
    true_one_hot = tf.one_hot(true_ragged, vocab_size, axis=-1)
    true_bags_of_words = tf.reduce_mean(true_one_hot, axis=-2)
    recon_all = tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(true_bags_of_words), pred_prob, axis=-1)
    recon = tf.reduce_mean(recon_all)
    return recon


def recon_loss_full(true, pred):
    true_slice = true[:, 1:]
    mask = tf.logical_not(tf.equal(true_slice, 0))
    recon_all = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.stop_gradient(true_slice), pred)
    recon_all_masked = tf.where(mask, x=recon_all, y=0.0)
    recon = tf.reduce_sum(recon_all_masked) / tf.reduce_sum(tf.cast(mask, 'float32'))
    return recon


def kld(dists_a, dists_b):
    return tf.reduce_mean(
        tfp.distributions.kl_divergence(
            dists_a,
            dists_b
        )
    )


def preg_loss(dists_a, dists_b):
    kld_a = kld(dists_a, dists_b)
    kld_b = kld(dists_b, dists_a)
    return kld_a, kld_b


def dists_means(dists_a, dists_b):
    mean_mean = (dists_a.mean() + dists_b.mean()) / 2
    mean_stddev = (dists_a.stddev() + dists_b.stddev()) / 2
    mean_dists = tfp.distributions.Normal(mean_mean, mean_stddev)
    return mean_dists


def mpreg_loss(dists_a, dists_b):
    mean_dists = dists_means(dists_a, dists_b)
    kld_a = kld(dists_a, mean_dists)
    kld_b = kld(dists_b, mean_dists)
    return kld_a, kld_b


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
        self.transformer_encoder = transformer.Encoder(4, 128, 8, 512, vocab_size, 1024, universal=True)
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
        self.vocab_size = vocab_size

    def call(self, inputs, training=None, **kwargs):
        return super(MlpBowDecoder, self).call(inputs, training=training)

    def recon_loss(self, true, pred):
        return recon_loss_bow(true, pred, self.vocab_size)


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

    def _single_bsd_step(self, preds_so_far, states):
        predicted_embedded = self.embedding(preds_so_far[:, -1])
        out, new_states = self.gru.cell(predicted_embedded, tf.expand_dims(states, 0))
        final = tf.nn.softmax(self.dense_2.layer(out), axis=-1)
        return final, new_states[0]

    def call(self, latent_samples, true_outputs=None, training=False, beam_width=1, **kwargs):
        if true_outputs is not None:
            return self.teacher_forcing_decode(latent_samples, true_outputs, training=training)
        else:
            dense_outs = self.dense(latent_samples)
            return beam_search_decode_new(dense_outs, self._single_bsd_step, self.start_token, self.end_token,
                                          beam_width=beam_width)

    def recon_loss(self, true, pred):
        return recon_loss_full(true, pred)


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
        self.transformer_decoder = transformer.Decoder(4, 128, 8, 512, vocab_size, reconstructed_dim, universal=True)
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

    def recon_loss(self, true, pred):
        return recon_loss_full(true, pred)


preprocessors = {
    'csharp': tdu.preprocess_csharp_or_java,
    'java': tdu.preprocess_csharp_or_java,
    'javadoc': tdu.preprocess_javadoc,
    'stackoverflow_query': tdu.preprocess_stackoverflow_summary,
    'edinburgh_python_or_summary': tdu.preprocess_edinburgh_python_or_summary
}

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

kld_losses = {
    'mpreg': mpreg_loss,
    'preg': preg_loss
}


class BimodalVariationalAutoEncoder(tf.Module):
    def __init__(self, model_path, train_set=None, val_set=None, num_train_epochs=0, train_batch_size=64,
                 sets_preprocessed=False, tf_name='bvae'):

        super(BimodalVariationalAutoEncoder, self).__init__(name=tf_name)

        model_path = os.path.abspath(model_path) + "/"
        if not os.path.isfile(model_path + "model_description.json"):
            raise FileNotFoundError("Model description not found")

        with open(model_path + "model_description.json", 'r') as json_file:
            model_description = json.load(json_file)

        self.language_preprocessor = preprocessors[model_description['l_type']]
        self.l_dim = model_description['l_dim']

        self.code_preprocessor = preprocessors[model_description['c_type']]
        self.c_dim = model_description['c_dim']

        self.language_tokenizer = Tokenizer(model_description['language_tokenizer_type'],
                                            model_path + model_description['language_tokenizer_path'],
                                            training_texts=([ex[0] for ex in train_set] if train_set is not None
                                                            else None),
                                            target_vocab_size=model_description['language_target_vocab_size'],
                                            vocab_min_count=model_description['language_vocab_min_count'])
        self.code_tokenizer = Tokenizer(model_description['code_tokenizer_type'],
                                        model_path + model_description['code_tokenizer_path'],
                                        training_texts=([ex[1] for ex in train_set] if train_set is not None
                                                        else None),
                                        target_vocab_size=model_description['code_target_vocab_size'],
                                        vocab_min_count=model_description['code_vocab_min_count'])

        self.latent_dim = model_description['latent_dim']
        self.kld_loss = kld_losses[model_description['kld_loss_type']]

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

        self.optimizer = optimizers[model_description['optimizer']]

        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_path + "checkpoints/", max_to_keep=3)
        if self.checkpoint_manager.latest_checkpoint:
            checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        if num_train_epochs > 0:
            self.train(train_set, val_set, num_epochs=num_train_epochs, batch_size=train_batch_size,
                       preprocessed=sets_preprocessed)

    @tf.function
    def loss(self, summaries, codes, al=0.35, bl=0.15, ac=0.35, bc=0.15):
        enc_language_dists = self.language_encoder(summaries, training=True)
        enc_source_code_dists = self.source_code_encoder(codes, training=True)
        enc_language = enc_language_dists.sample()
        enc_source_code = enc_source_code_dists.sample()
        dec_language = self.language_decoder(enc_language, training=True,
                                             true_outputs=summaries)
        dec_source_code = self.source_code_decoder(enc_source_code, training=True,
                                                   true_outputs=codes)
        language_kld, source_code_kld = self.kld_loss(enc_language_dists, enc_source_code_dists)
        language_recon = self.language_decoder.recon_loss(summaries, dec_language)
        source_code_recon = self.source_code_decoder.recon_loss(codes, dec_source_code)
        final_loss = al * language_recon + bl * language_kld + ac * source_code_recon + bc * source_code_kld
        return final_loss

    @tf.function
    def training_step(self, summaries, codes):
        with tf.GradientTape() as tape:
            loss = self.loss(summaries,
                             codes)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def evaluate(self, dataset, batch_size=64):
        dataset, num_batches = dataset_to_batched_tensors(dataset, batch_size, self.l_dim, self.c_dim)
        loss = 0.0
        batch_nums = tqdm.trange(num_batches)
        for i in batch_nums:
            summaries, codes = next(dataset)
            loss += self.loss(summaries, codes)
            if (i + 1) % 10 == 0:
                current_loss = loss / (i + 1)
                batch_nums.set_description("evaluate: loss=%.4f" % current_loss)
        loss /= num_batches
        return loss

    def train_one_epoch(self, train_set, batch_size=64):
        train_set, num_batches = dataset_to_batched_tensors(train_set, batch_size, self.l_dim, self.c_dim)
        train_loss = 0.0
        batch_nums = tqdm.trange(num_batches)
        for i in batch_nums:
            summaries, codes = next(train_set)
            train_loss += self.training_step(summaries, codes)
            if (i + 1) % 50 == 0:
                current_loss = train_loss / (i + 1)
                batch_nums.set_description("train: loss=%.4f" % current_loss)
        train_loss /= num_batches
        return train_loss

    def train(self, train_set, val_set, num_epochs=100, batch_size=64, patience=6, preprocessed=False):

        if not preprocessed:
            print("Preprocessing datasets...")
            train_set = [(self.language_preprocessor(s), self.code_preprocessor(c)) for s, c in train_set]
            val_set = [(self.language_preprocessor(s), self.code_preprocessor(c)) for s, c in val_set]

        print("Tokenizing datasets...")
        train_set = [(self.language_tokenizer.tokenize_text(s), self.code_tokenizer.tokenize_text(c))
                     for s, c in train_set]
        val_set = [(self.language_tokenizer.tokenize_text(s), self.code_tokenizer.tokenize_text(c))
                   for s, c in val_set]

        print("Removing examples that are too long...")
        num_train_before = len(train_set)
        train_set = [(s, c) for s, c in train_set if len(s) <= self.l_dim and len(c) <= self.c_dim]
        num_train = len(train_set)
        num_val_before = len(val_set)
        val_set = [(s, c) for s, c in val_set if len(s) <= self.l_dim and len(c) <= self.c_dim]
        num_val = len(val_set)
        print("%d training examples and %d val examples were left out" %
              (num_train_before - num_train, num_val_before - num_val))

        print("Training on %d examples, validating on %d examples" % (num_train, num_val))

        best_val_loss = self.evaluate(val_set, batch_size=batch_size)
        print("Initial validation loss: %.4f" % best_val_loss)
        num_epochs_with_no_improvement = 0

        for epoch in range(num_epochs):

            train_loss = self.train_one_epoch(train_set, batch_size=batch_size)
            val_loss = self.evaluate(val_set, batch_size=batch_size)

            print("Epoch %d of %d completed, train_loss=%.4f, val_loss=%.4f" %
                  (epoch + 1, num_epochs, train_loss, val_loss))

            if val_loss < best_val_loss:
                print("Val loss improved from %.4f, saving checkpoint" % best_val_loss)
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

    def summaries_to_latent(self, summaries, preprocessed=False):
        if not preprocessed:
            summaries = list(map(self.language_preprocessor, summaries))
        tokenized = list(map(self.language_tokenizer.tokenize_text, summaries))
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=self.l_dim, padding='post', value=0,
                                                               truncating='post')
        latent = self.language_encoder(padded, training=False)
        return latent

    def codes_to_latent(self, codes, preprocessed=False):
        if not preprocessed:
            codes = list(map(self.code_preprocessor, codes))
        tokenized = list(map(self.code_tokenizer.tokenize_text, codes))
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=self.c_dim, padding='post', value=0,
                                                               truncating='post')
        latent = self.source_code_encoder(padded, training=False)
        return latent

    def latent_to_summaries(self, latent, beam_width=1):
        mean = latent.mean()
        tokenized = self.language_decoder(mean, training=False, beam_width=beam_width)
        summaries = list(map(self.language_tokenizer.de_tokenize_text, tokenized))
        summaries = list(map(tdu.de_eof_text, summaries))
        return summaries

    def latent_to_codes(self, latent, beam_width=1):
        mean = latent.mean()
        tokenized = self.source_code_decoder(mean, training=False, beam_width=beam_width)
        codes = list(map(self.code_tokenizer.de_tokenize_text, tokenized))
        codes = list(map(tdu.de_eof_text, codes))
        return codes
