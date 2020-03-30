import sys
from bvae import *
from text_data_utils import *


assert len(sys.argv) == 2, "Usage: python train_bvae.py path/to/model/dir/"
model_path = sys.argv[1]

print("Loading dataset...")
train_summaries, train_codes = load_iyer_file("../data/iyer_csharp/train.txt")
val_summaries, val_codes = load_iyer_file("../data/iyer_csharp/valid.txt")

print("Creating model and seqifiers from JSON description...")
model = BimodalVariationalAutoEncoder(model_path, tokenizers_training_texts=(train_summaries, train_codes))

print("Starting training now...")
model.train(train_summaries, train_codes, val_summaries, val_codes)
