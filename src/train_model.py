import matplotlib.pyplot as plt
from bvae import *


def main():
    max_len = 40
    wv_size = 128
    language_wv, code_wv, train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes = \
        load_csv_dataset_with_w2v("../data2/processeed_data2.csv", max_len, wv_size)

    language_wv.save("saved_model/language_wv.txt")
    code_wv.save("saved_model/code_wv.txt")

    train_summaries = tokenized_texts_to_tensor(train_summaries, language_wv, max_len)
    val_summaries = tokenized_texts_to_tensor(val_summaries, language_wv, max_len)
    test_summaries = tokenized_texts_to_tensor(test_summaries, language_wv, max_len)

    train_codes = tokenized_texts_to_tensor(train_codes, code_wv, max_len)
    val_codes = tokenized_texts_to_tensor(val_codes, code_wv, max_len)
    test_codes = tokenized_texts_to_tensor(test_codes, code_wv, max_len)

    latent_dim = 128
    language_dim = train_summaries.shape[1]
    source_code_dim = train_codes.shape[1]

    model_description = {
        'wv_size': wv_size,
        'language_dim': language_dim,
        'source_code_dim': source_code_dim,
        'latent_dim': latent_dim,
    }
    with open("saved_model/model_description.json", 'w') as json_file:
        json.dump(model_description, json_file)

    model = BimodalVariationalAutoEncoder(language_dim, source_code_dim, latent_dim, wv_size, input_dropout=0.2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))

    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit((train_summaries, train_codes), None, batch_size=128, epochs=30,
                        validation_data=((val_summaries, val_codes), None),
                        callbacks=[reduce_on_plateau, early_stopping])

    model.save_weights("saved_model/model_weights")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss (mpreg) latent_dim=' + str(latent_dim))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    test_loss = model.evaluate((test_summaries, test_codes), None, batch_size=128, verbose=False)
    print("Test loss: " + str(test_loss))


if __name__ == "__main__":
    main()
