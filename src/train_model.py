import matplotlib.pyplot as plt
import random
from bvae import *


def train_bvae(model_path="../models/r4/", dataset_path="../data/iyer_csharp/",
               l_target_vocab_size=400):

    if not os.path.isfile(model_path + "model_description.json"):
        raise FileNotFoundError("Model description not found")

    with open(model_path + "model_description.json", 'r') as json_file:
        model_description = json.load(json_file)

    l_dim = model_description['l_dim']
    l_emb_dim = model_description['l_emb_dim']
    c_dim = model_description['c_dim']
    c_emb_dim = model_description['c_emb_dim']
    latent_dim = model_description['latent_dim']
    dropout_rate = model_description['dropout_rate']
    architecture = model_description['architecture']

    train_summaries, train_codes = load_iyer_file(dataset_path + "train.txt")
    val_summaries, val_codes = load_iyer_file(dataset_path + "valid.txt")
    test_summaries, test_codes = load_iyer_file(dataset_path + "test.txt")

    language_tokenizer_file = dataset_path + "language_tokenizer"
    if os.path.isfile(language_tokenizer_file + ".subwords"):
        language_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(language_tokenizer_file)
    else:
        language_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (summary for summary in train_summaries), l_target_vocab_size,
            reserved_tokens=['<s>', '</s>'])
        language_tokenizer.save_to_file(language_tokenizer_file)

    train_summaries = [language_tokenizer.encode(summary) for summary in train_summaries]
    val_summaries = [language_tokenizer.encode(summary) for summary in val_summaries]
    test_summaries = [language_tokenizer.encode(summary) for summary in test_summaries]

    train_codes = parse_codes(train_codes, c_dim)
    val_codes = parse_codes(val_codes, c_dim)
    test_codes = parse_codes(test_codes, c_dim)

    train_summaries, train_codes = trim_to_len(train_summaries, train_codes, l_dim, c_dim)
    val_summaries, val_codes = trim_to_len(val_summaries, val_codes, l_dim, c_dim)
    test_summaries, test_codes = trim_to_len(test_summaries, test_codes, l_dim, c_dim)

    c_vocab_size = MAX_LEXER_INDEX + 2

    model = BimodalVariationalAutoEncoder(l_dim, language_tokenizer.vocab_size, l_emb_dim,
                                          c_dim, c_vocab_size, c_emb_dim,
                                          latent_dim, input_dropout=dropout_rate, architecture=architecture)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=True)

    tf.keras.utils.plot_model(model, to_file=(model_path+'model_viz.png'), show_shapes=True, expand_nested=True)

    if os.path.isfile(model_path + "checkpoint"):
        model.load_weights(model_path + "model_checkpoint.ckpt")

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
    plt.title('model loss target_vocab_size=' + str(l_target_vocab_size) + 'latent_dim=' + str(latent_dim)
              + ' d=' + str(dropout_rate))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(model_path + 'performance_plot.png')

    test_loss = model.evaluate((test_summaries, test_codes), None, verbose=False)
    print("Test loss: " + str(test_loss))


if __name__ == "__main__":
    train_bvae()
