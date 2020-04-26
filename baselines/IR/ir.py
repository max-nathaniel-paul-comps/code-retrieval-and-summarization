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
    dists = np.array([textdistance.levenshtein(code, summaries_tok[i]) for i in range(len(summaries_tok))])
    min_dist_idx = int(np.argmin(dists))
    return summaries[min_dist_idx]


def test_ir_iyer():
    download('wordnet')
    dataset = tdu.load_iyer_dataset("../../data/iyer_csharp/dev.txt",
                                    alternate_summaries_filename="../../data/iyer_csharp/dev_alternate_summaries.txt")
    all_non_alt_summaries = [ex[0] for ex in dataset]
    refs = []
    preds = []
    meteors = []
    for i in range(len(dataset)):
        code = dataset[i][1]
        prediction = ir(code, all_non_alt_summaries)
        alt_summaries = dataset[i][2]
        print("Code: %s" % code)
        print("True summaries: %s" % alt_summaries)
        print("Predicted summary: %s" % prediction)
        meteor = meteor_score(alt_summaries, prediction)
        print("Sentence METEOR score: %.4f" % meteor)
        print()
        refs.append(alt_summaries)
        preds.append(prediction)
        meteors.append(meteor)
    meteor = np.mean(meteors)
    bleu = corpus_bleu(refs, preds)
    print("Average METEOR: ", meteor)
    print("Corpus BLEU-4: ", bleu)


if __name__ == "__main__":
    test_ir_iyer()
