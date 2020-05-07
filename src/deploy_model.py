import argparse
import tensorflow as tf
from bvae import BimodalVariationalAutoEncoder


parser = argparse.ArgumentParser(description="Deploy a BVAE to the SavedModel format for production use")
parser.add_argument("--model_path", help="Path to the input model", required=True)
parser.add_argument("--prod_model_path", help="Path to the output (production) model to create", required=True)

args = vars(parser.parse_args())
model_path = args["model_path"]
prod_model_path = args["prod_model_path"]

bvae = BimodalVariationalAutoEncoder(model_path)

tf.saved_model.save(bvae, prod_model_path)
