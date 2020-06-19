# Code Retrieval and Summarization
Our senior comps project for Carleton College.

This repository contains code for implementing, testing, analyzing, and applying different methods of code summarization
and retrieval.
- BVAE: Bimodal Variational AutoEncoder. An experimental
model architecture proposed by [Chen and Zhou (2018)](https://dl.acm.org/doi/10.1145/3238147.3240471)
that is used for both retrieval and summarization
- Baselines:
    - Retrieval: RET-IR
    - Summarization: IR
- Transformer (experimental, for summarization only)

Trained BVAE models in TensorFlow SavedModel format can be found in `models/`.
