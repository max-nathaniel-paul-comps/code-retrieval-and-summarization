import nltk
import numpy as np
import text_data_utils as tdu
import random
import argparse
from bvae import BimodalVariationalAutoEncoder
from transformer import CodeSummarizationTransformer


nltk.download('wordnet')

parser = argparse.ArgumentParser(description="Evaluate a model's source code summarization")
parser.add_argument("--prog_lang", help="What programming language to evaluate the summarization of",
                    choices=["csharp", "java", "python"], required=True)
parser.add_argument("--model_type", help="What kind of model you want to evaluate", choices=["bvae", "transformer"],
                    required=True)
parser.add_argument("--model_path", help="Path to the model", required=True)
args = vars(parser.parse_args())

model_type = args["model_type"]
model_path = args["model_path"]
prog_lang = args["prog_lang"]

if model_type == "bvae":
    bvae = BimodalVariationalAutoEncoder(model_path)
    summarize = lambda x: bvae.latent_to_summaries(bvae.codes_to_latent([x]))[0]
elif model_type == "transformer":
    transformer = CodeSummarizationTransformer(model_path)
    summarize = lambda x: transformer.summarize(x)
else:
    raise Exception()

if prog_lang == "csharp":
    dataset = tdu.load_iyer_dataset("../data/iyer_csharp/dev.txt",
                                    alternate_summaries_filename="../data/iyer_csharp/dev_alternate_summaries.txt")
    codes = [ex[1] for ex in dataset]
    true_summaries = [[ex[0]] + ex[2] for ex in dataset]
elif prog_lang == "python":
    _, _, dataset = tdu.load_edinburgh_dataset("../data/edinburgh_python")
    codes = [ex[1] for ex in dataset]
    true_summaries = [[ex[0]] for ex in dataset]
elif prog_lang == "java":
    dataset = tdu.load_json_dataset("../data/leclair_java/test.json")
    dataset = random.sample(dataset, 300)
    codes = [ex[1] for ex in dataset]
    true_summaries = [[ex[0]] for ex in dataset]
else:
    raise Exception()

meteors = []
predicts = []
for i in range(len(dataset)):
    print("%d of %d" % (i, len(dataset)))
    print("Code: %s" % codes[i])
    print("True Summaries: %s" % true_summaries[i])
    predicted = summarize(codes[i])
    print("%s Predicted Summary: %s" % (model_type, predicted))
    predicts.append(predicted)
    meteor = nltk.translate.meteor_score.meteor_score(true_summaries[i], predicted)
    print("%s METEOR score: %.4f" % (model_type, meteor))
    meteors.append(meteor)
    print()

average_meteor = np.mean(meteors)
print("%s Average METEOR score: %.4f\n" % (model_type, average_meteor))

corpus_bleu_4 = nltk.translate.bleu_score.corpus_bleu(true_summaries, predicts, weights=(0.25, 0.25, 0.25, 0.25))
corpus_bleu_2 = nltk.translate.bleu_score.corpus_bleu(true_summaries, predicts, weights=(0.5, 0.5, 0.0, 0.0))
print("%s Corpus BLEU-4 score: %.4f" % (model_type, corpus_bleu_4))
print("%s Corpus BLEU-2 score: %.4f" % (model_type, corpus_bleu_2))
