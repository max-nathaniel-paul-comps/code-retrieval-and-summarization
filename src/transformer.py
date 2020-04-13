import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
import sys

import text_data_utils as tdu
from tokenizer import Tokenizer


# This file was created from the TensorFlow Transformer tutorial
# It has been refactored to make it object-oriented, and some additional controls have been added to the training loop


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, universal=False):
        super(Encoder, self).__init__()

        self.universal = universal

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        if self.universal:
            self.enc_layer = EncoderLayer(d_model, num_heads, dff, rate)
        else:
            self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        if not self.universal:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            if self.universal:
                x += self.pos_encoding[:, :seq_len, :]
                x += self.pos_encoding[:, i, :]  # Timestep encoding
                x = self.enc_layer(x, training, mask)
            else:
                x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, universal=False):
        super(Decoder, self).__init__()

        self.universal = universal

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        if self.universal:
            self.dec_layer = DecoderLayer(d_model, num_heads, dff, rate)
        else:
            self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        if not self.universal:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            if self.universal:
                x += self.pos_encoding[:, :seq_len, :]
                x += self.pos_encoding[:, i, :]  # Timestep encoding
                x, block1, block2 = self.dec_layer(x, enc_output, training, look_ahead_mask, padding_mask)
            else:
                x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                       look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, model_path, input_tokenizer, output_tokenizer, rate=0.1,
                 universal=True, max_input_len=40, max_output_len=40):
        super(Transformer, self).__init__()

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate=rate, universal=universal)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate=rate, universal=universal)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        learning_rate = CustomSchedule(d_model)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        checkpoint_path = model_path + "train/"

        ckpt = tf.train.Checkpoint(transformer=self,
                                   optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ])
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.call(inp, tar_inp,
                                       True,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def val_loss(self, val_codes, val_summaries, batch_size=64):
        val_batches_per_epoch = int(val_codes.shape[0] / batch_size)
        val_loss = 0.0
        for batch in range(val_batches_per_epoch):
            inp = val_codes[batch * batch_size: batch * batch_size + batch_size]
            tar = val_summaries[batch * batch_size: batch * batch_size + batch_size]
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
            predictions, _ = self.call(inp, tar_inp,
                                       False,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
            loss = loss_function(tar_real, predictions)
            val_loss += loss
        val_loss /= val_batches_per_epoch
        return val_loss

    def train(self, train_codes, train_summaries, val_codes, val_summaries, batch_size=64, num_epochs=100):

        train_summaries = self.output_tokenizer.tokenize_texts(train_summaries)
        train_codes = self.input_tokenizer.tokenize_texts(train_codes)
        train_summaries, train_codes = tdu.sequences_to_tensors(train_summaries, train_codes, self.max_output_len,
                                                                self.max_input_len, dtype='int64')

        val_summaries = self.output_tokenizer.tokenize_texts(val_summaries)
        val_codes = self.input_tokenizer.tokenize_texts(val_codes)
        val_summaries, val_codes = tdu.sequences_to_tensors(val_summaries, val_codes, self.max_output_len,
                                                            self.max_input_len, dtype='int64')

        print("Training on %s samples, validating on %s samples." % (train_summaries.shape[0], val_summaries.shape[0]))

        batches_per_epoch = int(train_codes.shape[0] / batch_size)

        best_val_loss = self.val_loss(val_codes, val_summaries, batch_size=batch_size)
        print('Initial Validation loss: {:.4f}'.format(best_val_loss))
        num_epochs_with_no_improvement = 0

        for epoch in range(num_epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            batches = tqdm.trange(batches_per_epoch)
            for batch in batches:
                inp = train_codes[batch * batch_size: batch * batch_size + batch_size]
                tar = train_summaries[batch * batch_size: batch * batch_size + batch_size]

                self.train_step(inp, tar)

                if batch % 50 == 0:
                    batches.set_description("Epoch {} of {}, Loss {:.4f}, Accuracy {:.4f}".format(epoch + 1, num_epochs,
                                                                                            self.train_loss.result(),
                                                                                            self.train_accuracy.result()))

            val_loss = self.val_loss(val_codes, val_summaries, batch_size=batch_size)
            print('Validation loss: {:.4f}'.format(val_loss))

            if val_loss < best_val_loss:
                num_epochs_with_no_improvement = 0
                ckpt_save_path = self.ckpt_manager.save()
                best_val_loss = self.val_loss(val_codes, val_summaries, batch_size=batch_size)
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))
            else:
                num_epochs_with_no_improvement += 1
                print("Val loss did not improve")
                if num_epochs_with_no_improvement > 8:
                    print("Early stopping")
                    break

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                self.train_loss.result(),
                                                                self.train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def evaluate_on_sentence(self, inp_sentence, max_length):

        encoder_input = self.input_tokenizer.tokenize_texts([inp_sentence])
        if len(encoder_input[0]) > self.max_input_len:
            print("Warning: Input sentence exceeds maximum length")
        encoder_input = tf.keras.preprocessing.sequence.pad_sequences(encoder_input, maxlen=self.max_input_len,
                                                                      dtype='int64', padding='post', value=0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.output_tokenizer.start_token]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.call(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.output_tokenizer.end_token:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = self.input_tokenizer.tokenize_texts([sentence])[0]

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                [self.input_tokenizer.de_tokenize_texts([[i]])[0] for i in sentence],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.output_tokenizer.de_tokenize_texts([[i]]) for i in result
                                if i < self.output_tokenizer.vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()

    def translate(self, sentence, plot='', print_output=True):
        result, attention_weights = self.evaluate_on_sentence(sentence, self.max_output_len)

        predicted_sentence = self.output_tokenizer.de_tokenize_texts([result])[0]

        if print_output:
            print('Input: {}'.format(sentence))
            print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)

        return predicted_sentence

    def interactive_demo(self):
        while True:
            print()
            code = input(">> ")
            if code == "exit":
                break
            code = tdu.preprocess_source_code(code)
            self.translate(code)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class CodeSummarizationTransformer(object):
    def __init__(self, model_path, train=False, train_set=None, val_set=None):
        if train_set is not None:
            train_summaries = [ex[0] for ex in train_set]
            train_codes = [ex[1] for ex in train_set]
        else:
            train_summaries = None
            train_codes = None
        if val_set is not None:
            val_summaries = [ex[0] for ex in val_set]
            val_codes = [ex[1] for ex in val_set]
        else:
            val_summaries = None
            val_codes = None

        with open(model_path + "transformer_description.json") as transformer_desc_json:
            transformer_description = json.load(transformer_desc_json)

        language_tokenizer = Tokenizer(transformer_description['language_tokenizer_type'],
                                       model_path + transformer_description['language_tokenizer_path'],
                                       training_texts=train_summaries,
                                       target_vocab_size=transformer_description['language_target_vocab_size'])
        code_tokenizer = Tokenizer(transformer_description['code_tokenizer_type'],
                                   model_path + transformer_description['code_tokenizer_path'],
                                   training_texts=train_codes,
                                   target_vocab_size=transformer_description['code_target_vocab_size'])

        num_layers = transformer_description['num_layers']
        d_model = transformer_description['d_model']
        dff = transformer_description['dff']
        num_heads = transformer_description['num_heads']

        input_vocab_size = code_tokenizer.vocab_size
        target_vocab_size = language_tokenizer.vocab_size
        dropout_rate = transformer_description['dropout_rate']

        universal = transformer_description['universal']

        self.transformer = Transformer(num_layers, d_model, num_heads, dff,
                                       input_vocab_size, target_vocab_size,
                                       input_vocab_size, target_vocab_size,
                                       model_path, code_tokenizer, language_tokenizer,
                                       max_input_len=transformer_description['c_dim'],
                                       max_output_len=transformer_description['l_dim'],
                                       rate=dropout_rate, universal=universal)
        if train:
            self.transformer.train(train_codes, train_summaries, val_codes, val_summaries)


def main():
    assert len(sys.argv) == 3
    train = (sys.argv[1] == 'train')
    model_path = sys.argv[2]

    print("Loading dataset...")
    iyer_train = tdu.load_iyer_dataset("../data/iyer_csharp/train.txt")
    iyer_val = tdu.load_iyer_dataset("../data/iyer_csharp/valid.txt")
    our_train = tdu.load_csv_dataset("../data/our_csharp/train.csv")
    our_val = tdu.load_csv_dataset("../data/our_csharp/val.csv")
    all_train = list(set().union(iyer_train, our_train))
    all_val = list(set().union(iyer_val, our_val))

    print("Loading transformer...")
    transformer = CodeSummarizationTransformer(model_path, train=train, train_set=all_train, val_set=all_val)
    if not train:
        transformer.transformer.interactive_demo()


if __name__ == "__main__":
    main()
