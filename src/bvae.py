import tensorflow as tf
import tensorflow_probability as tfp
import gensim
import random
import matplotlib.pyplot as plt
from mlp import *
from text_data_utils import *


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


class BimodalVariationalAutoEncoder(tf.Module):
    def __init__(self, language_dim, source_code_dim, latent_dim, name='bvae'):
        super(BimodalVariationalAutoEncoder, self).__init__(name=name)
        self.language_encoder = VariationalEncoder(language_dim, latent_dim, 'language_encoder')
        self.source_code_encoder = VariationalEncoder(source_code_dim, latent_dim, 'source_code_encoder')
        self.language_decoder = Decoder(latent_dim, language_dim, 'language_decoder')
        self.source_code_decoder = Decoder(latent_dim, source_code_dim, 'source_code_decoder')

    def loss(self, language_batch, source_code_batch):
        enc_language_dists = self.language_encoder(language_batch)
        enc_source_code_dists = self.source_code_encoder(source_code_batch)
        mean_mean = (enc_language_dists.mean() + enc_source_code_dists.mean()) / 2
        mean_stddev = (enc_language_dists.stddev() + enc_source_code_dists.stddev()) / 2
        language_kl_divergence = tf.reduce_mean(
            tfp.distributions.kl_divergence(
                enc_language_dists,
                tfp.distributions.Normal(mean_mean, mean_stddev)
            )
        )
        source_code_kl_divergence = tf.reduce_mean(
            tfp.distributions.kl_divergence(
                enc_source_code_dists,
                tfp.distributions.Normal(mean_mean, mean_stddev)
            )
        )

        enc_language = enc_language_dists.sample()
        enc_source_code = enc_source_code_dists.sample()
        dec_language = self.language_decoder(enc_language)
        dec_source_code = self.source_code_decoder(enc_source_code)
        language_mse = tf.reduce_mean(tf.math.squared_difference(language_batch, dec_language))
        source_code_mse = tf.reduce_mean(tf.math.squared_difference(source_code_batch, dec_source_code))
        return language_kl_divergence + source_code_kl_divergence + language_mse + source_code_mse

    def training_step(self, language_batch, source_code_batch, optimizer):
        with tf.GradientTape() as t:
            current_loss = self.loss(language_batch, source_code_batch)
        grads = t.gradient(current_loss, self.trainable_variables)
        optimizer.apply_gradients((grads[i], self.trainable_variables[i]) for i in range(len(grads)))
        return current_loss

    def train(self, language_train_tensor, source_code_train_tensor, language_val_tensor, source_code_val_tensor,
              num_epochs, batch_size, optimizer):
        assert len(language_train_tensor) == len(source_code_train_tensor)
        for epoch_num in range(1, num_epochs + 1):
            train_losses = []
            for batch_num in range(int(len(language_train_tensor) / batch_size)):
                start = batch_num * batch_size
                end = batch_num * batch_size + batch_size
                language_batch = language_train_tensor[start: end]
                source_code_batch = source_code_train_tensor[start: end]
                current_loss = self.training_step(language_batch, source_code_batch, optimizer)
                train_losses.append(current_loss)
            train_loss = sum(train_losses) / len(train_losses)
            val_losses = []
            for batch_num in range(int(len(language_val_tensor) / batch_size)):
                start = batch_num * batch_size
                end = batch_num * batch_size + batch_size
                language_batch = language_val_tensor[start: end]
                source_code_batch = source_code_val_tensor[start: end]
                current_loss = self.loss(language_batch, source_code_batch)
                val_losses.append(current_loss)
            val_loss = sum(val_losses) / len(val_losses)
            print("Epoch {} of {} completed, training loss = {}, validation loss = {}".format(
                epoch_num, num_epochs, train_loss, val_loss))


def main():
    max_len = 55
    train_summaries, train_codes = load_iyer_file("../data/iyer/train.txt", max_len=max_len)
    summaries_wv = gensim.models.Word2Vec(train_summaries, size=100, min_count=5).wv
    codes_wv = gensim.models.Word2Vec(train_codes, size=100, min_count=5).wv
    train_summaries_tensor = tokenized_texts_to_tensor(train_summaries, summaries_wv, max_len)
    train_codes_tensor = tokenized_texts_to_tensor(train_codes, codes_wv, max_len)
    train_summaries_tensor_fl = np.reshape(train_summaries_tensor,
                                           (train_summaries_tensor.shape[0],
                                            train_summaries_tensor.shape[1] * train_summaries_tensor.shape[2]))
    train_codes_tensor_fl = np.reshape(train_codes_tensor,
                                       (train_codes_tensor.shape[0],
                                        train_codes_tensor.shape[1] * train_codes_tensor.shape[2]))

    val_summaries, val_codes = load_iyer_file("../data/iyer/valid.txt", max_len=max_len)
    val_summaries_tensor = tokenized_texts_to_tensor(val_summaries, summaries_wv, max_len)
    val_codes_tensor = tokenized_texts_to_tensor(val_codes, codes_wv, max_len)
    val_summaries_tensor_fl = np.reshape(val_summaries_tensor,
                                         (val_summaries_tensor.shape[0],
                                          val_summaries_tensor.shape[1] * val_summaries_tensor.shape[2]))
    val_codes_tensor_fl = np.reshape(val_codes_tensor,
                                     (val_codes_tensor.shape[0],
                                      val_codes_tensor.shape[1] * val_codes_tensor.shape[2]))

    test_summaries, test_codes = load_iyer_file("../data/iyer/test.txt", max_len=max_len)
    test_summaries_tensor = tokenized_texts_to_tensor(test_summaries, summaries_wv, max_len)
    test_codes_tensor = tokenized_texts_to_tensor(test_codes, codes_wv, max_len)
    test_summaries_tensor_fl = np.reshape(test_summaries_tensor,
                                          (test_summaries_tensor.shape[0],
                                           test_summaries_tensor.shape[1] * test_summaries_tensor.shape[2]))
    test_codes_tensor_fl = np.reshape(test_codes_tensor,
                                      (test_codes_tensor.shape[0],
                                       test_codes_tensor.shape[1] * test_codes_tensor.shape[2]))

    latent_dim = 768

    model = BimodalVariationalAutoEncoder(train_summaries_tensor_fl.shape[1],
                                          train_codes_tensor_fl.shape[1],
                                          latent_dim)

    model.train(train_summaries_tensor_fl, train_codes_tensor_fl, val_summaries_tensor_fl, val_codes_tensor_fl, 35, 128,
                tf.keras.optimizers.Adam(learning_rate=0.0001))

    random.seed()
    random_idx = random.randrange(test_summaries_tensor.shape[0])
    rand_test = np.array([test_summaries_tensor[random_idx]])
    print("(Test Set) Input: ", tensor_to_tokenized_texts(rand_test, summaries_wv)[0])
    rec = np.reshape(model.language_decoder(model.language_encoder(np.array([test_summaries_tensor_fl[random_idx]])).sample()), (1, test_summaries_tensor.shape[1], test_summaries_tensor.shape[2]))
    print("(Test Set) Reconstructed: ", tensor_to_tokenized_texts(rec, summaries_wv)[0])
    print("Hi")


if __name__ == "__main__":
    main()
