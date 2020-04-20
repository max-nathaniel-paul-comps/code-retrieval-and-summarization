import tensorflow_datasets as tfds
from transformer import Transformer
from tokenizer import Tokenizer


examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

train_pt = list(pt.numpy().decode("utf-8") for pt, en in train_examples)
train_en = list(en.numpy().decode("utf-8") for pt, en in train_examples)
val_pt = list(pt.numpy().decode("utf-8") for pt, en in val_examples)
val_en = list(en.numpy().decode("utf-8") for pt, en in val_examples)

model_dir = "../models/pt_to_en_tr_2/"

tokenizer_en = Tokenizer("subwords", model_dir + "english", target_vocab_size=2**13,
                         training_texts=train_en)
tokenizer_pt = Tokenizer("subwords", model_dir + "portugese", target_vocab_size=2**13,
                         training_texts=train_pt)

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1

max_input_len = 40
max_output_len = 40

transformer = Transformer(num_layers, d_model, num_heads, dff, tokenizer_pt.vocab_size,
                          tokenizer_en.vocab_size, max_input_len, max_output_len,
                          model_dir, tokenizer_pt, tokenizer_en, rate=dropout_rate, universal=True,
                          max_input_len=40, max_output_len=40, shared_qk=True)

# transformer.train(train_pt, train_en, val_pt, val_en, batch_size=64, num_epochs=100)

transformer.interactive_demo()
