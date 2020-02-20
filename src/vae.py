import tensorflow_probability as tfp
import random
import gensim
from text_data_utils import *
from mlp import *


class VariationalEncoder(MultiLayerPerceptron):
    def __init__(self, input_dim, latent_dim, wv_size, name='variational_encoder'):
        hidden_1_dim = int(input_dim * wv_size - (input_dim * wv_size - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(VariationalEncoder, self).__init__((input_dim, wv_size), 2 * latent_dim, [hidden_1_dim, hidden_2_dim], name=name)
        self.latent_dim = latent_dim

    def __call__(self, x):
        dist = super(VariationalEncoder, self).__call__(x)
        mean = dist[:, :self.latent_dim]
        stddev = tf.math.abs(dist[:, self.latent_dim:])
        return tfp.distributions.Normal(mean, stddev)


class Decoder(MultiLayerPerceptron):
    def __init__(self, latent_dim, reconstructed_dim, wv_size, name='decoder'):
        hidden_1_dim = int(reconstructed_dim * wv_size - (reconstructed_dim * wv_size - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(Decoder, self).__init__(latent_dim, (reconstructed_dim, wv_size), [hidden_2_dim, hidden_1_dim], name=name)


class VariationalAutoEncoder(tf.Module):
    def __init__(self, input_dim, latent_dim, wv_size, name='vae'):
        super(VariationalAutoEncoder, self).__init__(name=name)
        self.encoder = VariationalEncoder(input_dim, latent_dim, wv_size)
        self.decoder = Decoder(latent_dim, input_dim, wv_size)
        self.wv_size = wv_size

    def loss(self, inputs):
        latent_dists = self.encoder(inputs)
        sample_latent = latent_dists.sample()
        emp_dist = tfp.distributions.Empirical(sample_latent, event_ndims=1)
        kl_divergence = tf.reduce_mean(
            tfp.distributions.kl_divergence(
                tfp.distributions.Normal(emp_dist.mean(), emp_dist.stddev()),
                tfp.distributions.Normal(0.0, 1.0)
            )
        )
        decoded = self.decoder(sample_latent)
        mask = tf.reduce_all(tf.equal(inputs, 0.0), axis=-1)
        recon_tensor = tf.losses.cosine_similarity(inputs, decoded) + 1
        recon_masked = tf.where(mask, x=0.0, y=recon_tensor)
        recon = tf.reduce_sum(recon_masked) / tf.reduce_sum(tf.cast(mask, 'float32'))
        return kl_divergence + recon

    def training_step(self, inputs, optimizer):
        with tf.GradientTape() as t:
            current_loss = self.loss(inputs)
        grads = t.gradient(current_loss, self.trainable_variables)
        optimizer.apply_gradients((grads[i], self.trainable_variables[i]) for i in range(len(grads)))
        return current_loss

    def train(self, inputs, val_inputs, num_epochs, batch_size, optimizer):
        for epoch_num in range(1, num_epochs + 1):
            train_losses = []
            for batch_num in range(int(len(inputs) / batch_size)):
                start = batch_num * batch_size
                end = batch_num * batch_size + batch_size
                batch = inputs[start: end]
                current_loss = self.training_step(batch, optimizer)
                train_losses.append(current_loss)
            train_loss = sum(train_losses) / len(train_losses)
            val_losses = []
            for batch_num in range(int(len(val_inputs) / batch_size)):
                start = batch_num * batch_size
                end = batch_num * batch_size + batch_size
                batch = val_inputs[start: end]
                current_loss = self.loss(batch)
                val_losses.append(current_loss)
            val_loss = sum(val_losses) / len(val_losses)
            print("Epoch {} of {} completed, training loss = {}, validation loss = {}".format(
                epoch_num, num_epochs, train_loss, val_loss))


def main():
    language_wv = gensim.models.KeyedVectors.load_word2vec_format("../data/embeddings/w2v_format_summaries_vectors.txt")
    wv_size = language_wv.vector_size

    train_summaries, train_codes = load_iyer_file("../data/iyer_csharp/train.txt")
    val_summaries, val_codes = load_iyer_file("../data/iyer_csharp/valid.txt")
    test_summaries, test_codes = load_iyer_file("../data/iyer_csharp/test.txt")

    max_len = max(len(summary) for summary in train_summaries)

    train_summaries = tokenized_texts_to_tensor(train_summaries, language_wv, max_len)
    val_summaries = tokenized_texts_to_tensor(val_summaries, language_wv, max_len)
    test_summaries = tokenized_texts_to_tensor(test_summaries, language_wv, max_len)

    model = VariationalAutoEncoder(train_summaries.shape[1], 256, wv_size)
    model.train(train_summaries, val_summaries, 6, 128, tf.keras.optimizers.Adam(learning_rate=0.001))

    for _ in range(20):
        random.seed()
        random_idx = random.randrange(test_summaries.shape[0])
        rand_test = np.array([test_summaries[random_idx]])
        print("(Test Set) Input: ", tensor_to_tokenized_texts(rand_test, language_wv)[0])
        rec = model.decoder(model.encoder(rand_test).mean()).numpy()
        print("(Test Set) Recon: ", tensor_to_tokenized_texts(rec, language_wv)[0])
        print()


if __name__ == "__main__":
    main()
