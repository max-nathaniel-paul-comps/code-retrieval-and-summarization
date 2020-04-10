import sys
import bvae
import text_data_utils as tdu


assert len(sys.argv) == 3, "Usage: python train_bvae.py num_epochs path/to/model/dir/"
num_epochs = int(sys.argv[1])
model_path = sys.argv[2]

print("Loading dataset...")
iyer_train = tdu.load_iyer_dataset("../data/iyer_csharp/train.txt")
iyer_val = tdu.load_iyer_dataset("../data/iyer_csharp/valid.txt")
our_train = tdu.load_csv_dataset("../data/our_csharp/train.csv")
our_val = tdu.load_csv_dataset("../data/our_csharp/val.csv")
train = list(set().union(iyer_train, our_train))
val = list(set().union(iyer_val, our_val))

print("Creating model and tokenizers from JSON description...")
model = bvae.BimodalVariationalAutoEncoder(model_path, tokenizers_training_texts=train)

print("Starting training now...")
model.train(train, val, num_epochs=num_epochs)
