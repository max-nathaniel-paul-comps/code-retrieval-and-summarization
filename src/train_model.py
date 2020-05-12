import argparse
import bvae
import text_data_utils as tdu
from transformer import Transformer


parser = argparse.ArgumentParser(description="Train a BVAE or Transformer")
parser.add_argument("--model_type", help="What type of model to train", required=True, choices=["bvae", "transformer"])
parser.add_argument("--prog_lang", help="What programming language to train on", required=True,
                    choices=["csharp", "python", "java"])
parser.add_argument("--num_epochs", help="Number of epochs to train the model", required=True, type=int)
parser.add_argument("--model_path", help="Path to BVAE or Transformer model", required=True, type=str)
args = vars(parser.parse_args())
model_type = args["model_type"]
prog_lang = args["prog_lang"]
model_path = args["model_path"]
num_epochs = args["num_epochs"]

print("Loading dataset...")
if prog_lang == "csharp":
    iyer_train = tdu.load_iyer_dataset("../data/iyer_csharp/train.txt")
    iyer_val = tdu.load_iyer_dataset("../data/iyer_csharp/valid.txt")
    our_train = tdu.load_csv_dataset("../data/our_csharp/train.csv")
    our_val = tdu.load_csv_dataset("../data/our_csharp/val.csv")
    all_train = list(set().union(iyer_train, our_train))
    all_val = list(set().union(iyer_val, our_val))
elif prog_lang == "python":
    all_train = list(tdu.edinburgh_dataset_as_generator("../data/edinburgh_python/data_ps.descriptions.train.txt",
                                                        "../data/edinburgh_python/data_ps.declbodies.train.txt")())
    all_val = list(tdu.edinburgh_dataset_as_generator("../data/edinburgh_python/data_ps.descriptions.valid.txt",
                                                      "../data/edinburgh_python/data_ps.declbodies.valid.txt")())
elif prog_lang == "java":
    all_train = tdu.load_json_dataset("../data/leclair_java/train.json")
    all_val = tdu.load_json_dataset("../data/leclair_java/val.json")
else:
    raise Exception()

if model_type == "bvae":
    model = bvae.BimodalVariationalAutoEncoder(model_path, train_set=all_train, val_set=all_val,
                                               num_train_epochs=num_epochs, sets_preprocessed=True)
elif model_type == "transformer":
    model = Transformer(model_path, train_set=all_train, val_set=all_val, num_train_epochs=num_epochs,
                        sets_preprocessed=True)
else:
    raise Exception()
