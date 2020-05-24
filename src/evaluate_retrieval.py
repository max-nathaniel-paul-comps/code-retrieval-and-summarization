import random
import numpy as np
import text_data_utils as tdu
import argparse
import tqdm
import sys
from bvae import BimodalVariationalAutoEncoder
from ret_bvae import RetBVAE
sys.path.append("../baselines/RET-IR")
import retir


def reciprocal_rank(sorted_indices, golden_idx):
    for i in range(len(sorted_indices)):
        if sorted_indices[i] == golden_idx:
            return 1.0 / (i + 1.0)


def random_samples_method(dataset, bvae_model, compare_ret_ir, num_random_samples, random_sample_size):
    """
    The method used by Chen and Zhou.
    """
    random_reciprocal_ranks = []
    ret_ir_reciprocal_ranks = []
    bvae_reciprocal_ranks = []
    for _ in tqdm.tqdm(range(num_random_samples)):
        rand_idx = random.randrange(0, len(dataset) - random_sample_size)
        rand_samples = dataset[rand_idx: rand_idx + random_sample_size]

        rand_summaries = [ex[0] for ex in rand_samples]
        rand_codes = [ex[1] for ex in rand_samples]

        golden_idx = random.randrange(random_sample_size)
        rand_alt_summary_idx = random.randrange(len(rand_samples[golden_idx][2]))
        rand_alt_summary = rand_samples[golden_idx][2][rand_alt_summary_idx]

        random_sorted_indices = random.sample(range(random_sample_size), random_sample_size)
        random_reciprocal_ranks.append(reciprocal_rank(random_sorted_indices, golden_idx))
        if compare_ret_ir:
            ret_ir_sorted_indices = retir.retir(rand_alt_summary, rand_summaries, random_sample_size)
            ret_ir_reciprocal_ranks.append(reciprocal_rank(ret_ir_sorted_indices, golden_idx))

        bvae_retriever = RetBVAE(bvae_model, rand_codes)
        bvae_sorted_indices = bvae_retriever.rank_options(rand_alt_summary)
        bvae_reciprocal_ranks.append(reciprocal_rank(bvae_sorted_indices, golden_idx))

    random_mrr = np.mean(random_reciprocal_ranks)
    if compare_ret_ir:
        ret_ir_mrr = np.mean(ret_ir_reciprocal_ranks)
    else:
        ret_ir_mrr = None
    bvae_mrr = np.mean(bvae_reciprocal_ranks)

    return random_mrr, ret_ir_mrr, bvae_mrr


def full_method(dataset, bvae_model, compare_ret_ir):
    """
    Instead of taking random samples, simply evaluate on the entire target dataset.
    Tends to produce less impressive results because there are more distractors on each query.
    """
    summaries = [ex[0] for ex in dataset]
    codes = [ex[1] for ex in dataset]
    alternate_summaries = [ex[2] for ex in dataset]

    retriever = RetBVAE(bvae_model, codes)

    random_reciprocal_ranks = []
    ret_ir_reciprocal_ranks = []
    bvae_reciprocal_ranks = []

    for i in tqdm.trange(len(dataset)):
        for summary in alternate_summaries[i]:

            random_sorted_indices = random.sample(range(len(dataset)), len(dataset))
            random_reciprocal_ranks.append(reciprocal_rank(random_sorted_indices, i))
            if compare_ret_ir:
                ret_ir_sorted_indices = retir.retir(summary, summaries, len(dataset))
                ret_ir_reciprocal_ranks.append(reciprocal_rank(ret_ir_sorted_indices, i))

            bvae_sorted_indices = retriever.rank_options(summary)
            bvae_reciprocal_ranks.append(reciprocal_rank(bvae_sorted_indices, i))

    random_mrr = np.mean(random_reciprocal_ranks)
    if compare_ret_ir:
        ret_ir_mrr = np.mean(ret_ir_reciprocal_ranks)
    else:
        ret_ir_mrr = None
    bvae_mrr = np.mean(bvae_reciprocal_ranks)

    return random_mrr, ret_ir_mrr, bvae_mrr


def main():
    parser = argparse.ArgumentParser(description="Evaluate code retrieval")
    parser.add_argument("--prog_lang", help="What programming language to evaluate on",
                        choices=["csharp", "java", "python"], required=True)
    parser.add_argument("--bvae_model_path", help="Path to the BVAE", required=True)
    parser.add_argument("--method", help="Evaluation method", choices=["random_samples", "full"],
                        default="random_samples")
    parser.add_argument("--num_random_samples", help="Number of random samples to take. Only valid if `random_samples`"
                                                     "evaluation method is chosen.",
                        default=10000, type=int)
    parser.add_argument("--random_sample_size", help="Size of random samples to take. Only valid if `random_samples`"
                                                     "evaluation method is chosen.",
                        default=50, type=int)
    parser.add_argument("--compare_ret_ir", help="Whether to compare RET-IR to the BVAE and random guess",
                        choices=[0, 1], default=0, type=int)
    args = vars(parser.parse_args())
    prog_lang = args["prog_lang"]
    bvae_model_path = args["bvae_model_path"]
    method = args["method"]
    num_random_samples = args["num_random_samples"]
    random_sample_size = args["random_sample_size"]
    compare_ret_ir = args["compare_ret_ir"]

    if prog_lang == "csharp":
        dataset = tdu.load_iyer_dataset("../data/iyer_csharp/eval.txt",
                                        alternate_summaries_filename="../data/iyer_csharp/eval_alternate_summaries.txt")
    elif prog_lang == "python":
        _, _, test = tdu.load_edinburgh_dataset("../data/edinburgh_python")
        dataset = [("I'M IN THE ALT", ex[1], [ex[0]]) for ex in test]
    elif prog_lang == "java":
        dataset = tdu.load_json_dataset("../data/leclair_java/test.json")
        dataset = [("I'M IN THE ALT", ex[1], [ex[0]]) for ex in dataset]
    else:
        raise Exception()

    bvae_model = BimodalVariationalAutoEncoder(bvae_model_path)

    random.seed()
    if method == 'random_samples':
        random_mrr, ret_ir_mrr, bvae_mrr = random_samples_method(dataset, bvae_model, compare_ret_ir,
                                                                 num_random_samples, random_sample_size)
    elif method == 'full':
        random_mrr, ret_ir_mrr, bvae_mrr = full_method(dataset, bvae_model, compare_ret_ir)
    else:
        raise Exception()

    print("Random Choice Mean Reciprocal Rank: %.4f" % random_mrr)
    if compare_ret_ir:
        print("RET-IR Mean Reciprocal Rank: %.4f" % ret_ir_mrr)
    print("BVAE Mean Reciprocal Rank: %.4f" % bvae_mrr)


if __name__ == "__main__":
    main()
