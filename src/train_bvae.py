from bvae import *


def train_bvae(model, model_path, train_summaries, train_codes, val_summaries, val_codes):

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.abspath(model_path + "tboard"), histogram_freq=1)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(model_path + 'model_checkpoint.ckpt',
                                                     verbose=True, save_best_only=True,
                                                     monitor='val_loss', save_freq='epoch')
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit((train_summaries, train_codes), None, batch_size=128, epochs=100,
                        validation_data=((val_summaries, val_codes), None),
                        callbacks=[tboard_callback, checkpoints, reduce_on_plateau, early_stopping])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(model_path + 'performance_plot.png')


def main(model_path="../models/r8/"):
    print("Loading dataset...")
    train_summaries, train_codes = load_iyer_file("../data/iyer_csharp/train.txt")
    val_summaries, val_codes = load_iyer_file("../data/iyer_csharp/valid.txt")

    print("Creating model from JSON description...")
    model = load_or_create_model(model_path)

    print("Loading seqifiers, which are responsible for turning texts into sequences of integers...")
    language_seqifier = load_or_create_seqifier(model_path + "language_seqifier.json",
                                                model.l_vocab_size,
                                                training_texts=train_summaries,
                                                tokenization=lambda s: tokenize_texts(s))
    code_seqifier = load_or_create_seqifier(model_path + "code_seqifier.json",
                                            model.c_vocab_size,
                                            training_texts=train_codes,
                                            tokenization=lambda c: parse_codes(c, model.c_dim))

    if os.path.isfile(model_path + "checkpoint"):
        print("The model has already been trained, and training will continue.")
    else:
        print("The model has not been trained yet.")

    print("Preparing datasets for training...")
    train_summaries, train_codes = process_dataset(train_summaries, train_codes, language_seqifier, code_seqifier,
                                                   model.l_dim, model.c_dim)
    val_summaries, val_codes = process_dataset(val_summaries, val_codes, language_seqifier, code_seqifier,
                                               model.l_dim, model.c_dim)

    print("Starting training now...")
    train_bvae(model, model_path, train_summaries, train_codes, val_summaries, val_codes)


if __name__ == "__main__":
    main()
