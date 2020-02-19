import gensim
import random
import matplotlib.pyplot as plt
from vae import *
from text_data_utils import *


class BimodalVariationalAutoEncoder(tf.Module):
    def __init__(self, language_dim, source_code_dim, latent_dim, wv_size, name='bvae'):
        super(BimodalVariationalAutoEncoder, self).__init__(name=name)
        self.language_encoder = VariationalEncoder(language_dim, latent_dim, wv_size, name='language_encoder')
        self.source_code_encoder = VariationalEncoder(source_code_dim, latent_dim, wv_size, name='source_code_encoder')
        self.language_decoder = Decoder(latent_dim, language_dim, wv_size, name='language_decoder')
        self.source_code_decoder = Decoder(latent_dim, source_code_dim, wv_size, name='source_code_decoder')

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

        language_recon = tf.reduce_mean(tf.losses.cosine_similarity(language_batch, dec_language) + 1)
        source_code_recon = tf.reduce_mean(tf.losses.cosine_similarity(source_code_batch, dec_source_code) + 1)

        return language_kl_divergence + source_code_kl_divergence + language_recon + source_code_recon

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
    language_wv = gensim.models.KeyedVectors.load_word2vec_format("../data/embeddings/w2v_format_summaries_vectors.txt")
    code_wv = gensim.models.KeyedVectors.load_word2vec_format("../data/embeddings/w2v_format_codes_vectors.txt")
    assert language_wv.vector_size == code_wv.vector_size
    wv_size = language_wv.vector_size

    max_len = 39
    train_summaries, train_codes = load_iyer_file("../data/iyer_csharp/train.txt", max_len=max_len)
    val_summaries, val_codes = load_iyer_file("../data/iyer_csharp/valid.txt", max_len=max_len)
    test_summaries, test_codes = load_iyer_file("../data/iyer_csharp/test.txt", max_len=max_len)

    train_summaries = tokenized_texts_to_tensor(train_summaries, language_wv, max_len)
    val_summaries = tokenized_texts_to_tensor(val_summaries, language_wv, max_len)
    test_summaries = tokenized_texts_to_tensor(test_summaries, language_wv, max_len)

    train_codes = tokenized_texts_to_tensor(train_codes, code_wv, max_len)
    val_codes = tokenized_texts_to_tensor(val_codes, code_wv, max_len)
    test_codes = tokenized_texts_to_tensor(test_codes, code_wv, max_len)

    latent_dim = 768

    model = BimodalVariationalAutoEncoder(train_summaries.shape[1], train_codes.shape[1], latent_dim, wv_size)

    model.train(train_summaries, train_codes, val_summaries, val_codes, 35, 128,
                tf.keras.optimizers.Adam(learning_rate=0.0001))

    for _ in range(20):
        random.seed()
        random_idx = random.randrange(test_summaries.shape[0])
        rand_test_summary = np.array([test_summaries[random_idx]])
        print("(Test Set) Input Summary: ", tensor_to_tokenized_texts(rand_test_summary, language_wv)[0])
        rand_test_code = np.array([test_codes[random_idx]])
        print("(Test Set) Input Code: ", tensor_to_tokenized_texts(rand_test_code, code_wv)[0])
        rec_summary = model.language_decoder(model.language_encoder(rand_test_summary).mean()).numpy()
        print("(Test Set) Reconstructed Summary: ", tensor_to_tokenized_texts(rec_summary, language_wv)[0])
        rec_code = model.source_code_decoder(model.source_code_encoder(rand_test_code).mean()).numpy()
        print("(Test Set) Reconstructed Source Code: ", tensor_to_tokenized_texts(rec_code, code_wv)[0])
        rec_summary_hard = model.language_decoder(model.source_code_encoder(rand_test_code).mean()).numpy()
        print("(Test Set) Reconstructed Summary From Source Code: ", tensor_to_tokenized_texts(rec_summary_hard, language_wv)[0])
        rec_code_hard = model.source_code_decoder(model.language_encoder(rand_test_summary).mean()).numpy()
        print("(Test Set) Reconstructed Source Code From Summary: ", tensor_to_tokenized_texts(rec_code_hard, code_wv)[0])
        print()


if __name__ == "__main__":
    main()
