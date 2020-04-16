import sys
import bvae
import text_data_utils as tdu


assert len(sys.argv) == 4, "Usage: python train_bvae.py num_epochs dataset_name path/to/model/dir/"
num_epochs = int(sys.argv[1])
dataset_name = sys.argv[2]
model_path = sys.argv[3]

print("Loading dataset...")
if dataset_name == "csharp":
    iyer_train = tdu.load_iyer_dataset("../data/iyer_csharp/train.txt")
    iyer_val = tdu.load_iyer_dataset("../data/iyer_csharp/valid.txt")
    our_train = tdu.load_csv_dataset("../data/our_csharp/train.csv")
    our_val = tdu.load_csv_dataset("../data/our_csharp/val.csv")
    train = list(set().union(iyer_train, our_train))
    val = list(set().union(iyer_val, our_val))
elif dataset_name == "python":
    train, val, _ = tdu.load_edinburgh_dataset("../data/edinburgh_python")
else:
    raise Exception("Invalid dataset_name: %s" % dataset_name)

print("Creating model and tokenizers from JSON description...")
model = bvae.BimodalVariationalAutoEncoder(model_path, tokenizers_training_texts=train)

print("Starting training now...")
model.train(train, val, num_epochs=num_epochs)
