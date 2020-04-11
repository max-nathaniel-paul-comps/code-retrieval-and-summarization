import random
import numpy as np
from ret_bvae import *
import sys
sys.path.append("../baselines/RET-IR")
import retir


def reciprocal_rank(sorted_indices, golden_idx):
    for i in range(len(sorted_indices)):
        if sorted_indices[i] == golden_idx:
            return 1.0 / (i + 1.0)


def random_samples_method(dataset, bvae_model, baseline_model, random_sample_size=50, num_samples=1000):
    """
    The method used by Chen and Zhou.
    """
    random.seed()
    baseline_reciprocal_ranks = []
    bvae_reciprocal_ranks = []
    for _ in tqdm.tqdm(range(num_samples)):
        rand_idx = random.randrange(0, len(dataset) - random_sample_size)
        rand_samples = dataset[rand_idx: rand_idx + random_sample_size]

        rand_summaries = [ex[0] for ex in rand_samples]
        rand_codes = [ex[1] for ex in rand_samples]

        golden_idx = random.randrange(random_sample_size)
        rand_alt_summary_idx = random.randrange(len(rand_samples[golden_idx][2]))
        rand_alt_summary = rand_samples[golden_idx][2][rand_alt_summary_idx]

        baseline_sorted_indices = baseline_model(rand_alt_summary, rand_summaries, rand_codes, random_sample_size)
        baseline_reciprocal_ranks.append(reciprocal_rank(baseline_sorted_indices, golden_idx))

        retriever = RetBVAE(bvae_model, rand_codes)
        bvae_sorted_indices = retriever.rank_options(rand_alt_summary)
        bvae_reciprocal_ranks.append(reciprocal_rank(bvae_sorted_indices, golden_idx))

    baseline_mean_reciprocal_rank = np.mean(baseline_reciprocal_ranks)
    bvae_mean_reciprocal_rank = np.mean(bvae_reciprocal_ranks)

    return baseline_mean_reciprocal_rank, bvae_mean_reciprocal_rank


def full_method(dataset, bvae_model, baseline_model):
    """
    Instead of taking random samples, simply evaluate on the entire target dataset.
    Tends to produce less impressive results because there are more distractors on each query.
    """
    summaries = [ex[0] for ex in dataset]
    codes = [ex[1] for ex in dataset]
    alternate_summaries = [ex[2] for ex in dataset]

    retriever = RetBVAE(bvae_model, codes)

    baseline_reciprocal_ranks = []
    bvae_reciprocal_ranks = []

    for i in tqdm.trange(len(dataset)):
        for summary in alternate_summaries[i]:

            baseline_sorted_indices = baseline_model(summary, summaries, codes, len(dataset))
            baseline_reciprocal_ranks.append(reciprocal_rank(baseline_sorted_indices, i))

            bvae_sorted_indices = retriever.rank_options(summary)
            bvae_reciprocal_ranks.append(reciprocal_rank(bvae_sorted_indices, i))

    baseline_mean_reciprocal_rank = np.mean(baseline_reciprocal_ranks)
    bvae_mean_reciprocal_rank = np.mean(bvae_reciprocal_ranks)

    return baseline_mean_reciprocal_rank, bvae_mean_reciprocal_rank


def main():
    assert len(sys.argv) == 4, "Usage: python evaluate_retrieval.py method baseline_name path/to/bvae/model/dir/"
    method = sys.argv[1]
    baseline = sys.argv[2]
    bvae_model_path = sys.argv[3]

    if baseline == 'ret_ir':
        def baseline_model(summary, candidate_summaries, candidate_codes, sample_size):
            # Calling the version for pretokenized because ret-ir performs better at character-level for some reason
            return retir.retir(summary, candidate_summaries, sample_size)

    elif baseline == 'random':
        def baseline_model(summary, candidate_summaries, candidate_codes, sample_size):
            return random.sample(range(len(candidate_summaries)), len(candidate_summaries))

    else:
        raise Exception("Invalid baseline specified: %s" % baseline)

    dataset = load_iyer_dataset("../data/iyer_csharp/dev.txt",
                                alternate_summaries_filename="../data/iyer_csharp/dev_alternate_summaries.txt")

    bvae_model = BimodalVariationalAutoEncoder(bvae_model_path)

    if method == 'random_samples':
        baseline_mrr, bvae_mrr = random_samples_method(dataset, bvae_model, baseline_model)
    elif method == 'full':
        baseline_mrr, bvae_mrr = full_method(dataset, bvae_model, baseline_model)
    else:
        raise Exception("Invalid method specified: %s" % method)

    print("Baseline Mean Reciprocal Rank: %s" % baseline_mrr)
    print("BVAE Mean Reciprocal Rank: %s" % bvae_mrr)


if __name__ == "__main__":
    main()
