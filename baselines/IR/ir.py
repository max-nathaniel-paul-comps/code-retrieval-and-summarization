import textdistance
import sys
from nltk import download
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np
sys.path.append('../../src')
import text_data_utils as tdu


def ir(code, summaries):
    code = tdu.tokenize_text(code)
    summaries_tok = tdu.tokenize_texts(summaries)
    print("post-tokenize")
    dists = np.array([textdistance.levenshtein(code, summaries_tok[i]) for i in range(len(summaries_tok))])
    print("post-dists")
    min_dist_idx = int(np.argmin(dists))
    return summaries[min_dist_idx]


def test_ir_iyer(ir_use_alt_summaries=True):
    download('wordnet')
    #dataset = tdu.load_iyer_dataset("../../data/iyer_csharp/dev.txt",
                                    #alternate_summaries_filename="../../data/iyer_csharp/dev_alternate_summaries.txt")
    print("got here")
    #dataset = tdu.load_edinburgh_dataset("../../data/edinburgh_python")[2]
    dataset = tdu.load_json_dataset("../../data/leclair_java/test.json")
    print("loaded")
    summaries_for_ir_to_choose_from = [dataset[ex][0] for ex in range(100)]
    if ir_use_alt_summaries:
        for ex in dataset:
            for alt in ex[2]:
                summaries_for_ir_to_choose_from.append(alt)
    refs = []
    preds = []
    meteors = []
    for i in range(100):
        print("started loop")
        code = dataset[i][1]
        print(code)
        print("pre-ir")
        prediction = ir(code, summaries_for_ir_to_choose_from)
        print("post-ir")
        true_summaries = dataset[i][0]
        if ir_use_alt_summaries:
            true_summaries.append(dataset[i][0])
        print("Code: %s" % code)
        print("True summaries: %s" % true_summaries)
        print("Predicted summary: %s" % prediction)
        meteor = meteor_score(true_summaries, prediction)
        print("Sentence METEOR score: %.4f" % meteor)
        print()
        refs.append(true_summaries)
        preds.append(prediction)
        meteors.append(meteor)
    meteor = np.mean(meteors)
    bleu = corpus_bleu(refs, preds)
    print("Average METEOR: ", meteor)
    print("Corpus BLEU-4: ", bleu)


def main():
    ir_use_alt_summaries = input("Should IR use the alternate summaries? (yes/no) ")
    if ir_use_alt_summaries == "yes":
        test_ir_iyer(ir_use_alt_summaries=True)
    elif ir_use_alt_summaries == "no":
        test_ir_iyer(ir_use_alt_summaries=False)
    else:
        raise Exception("lmao")


if __name__ == "__main__":
    main()
