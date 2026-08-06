# Sparse Routing for Long-Context Retrieval

## 1 Introduction

Long-context transformers degrade on retrieval-heavy workloads. We introduce SparseRoute, a
router that selects 4 of 32 expert blocks per token. Prior work [1] reported 61.2 accuracy on
the same benchmark. Our method reaches 74.8 accuracy, described further in Section 4.

## 2 Method

### 2.1 Router

The router scores each expert with a bilinear form and keeps the top-k experts, as in Eq. 1.

$$
s_i = x^\top W_r e_i + b_i \quad (1)
$$

We set k = 4 throughout. The routing temperature is 0.7 and the load-balancing coefficient is 0.01.

### 2.2 Training

We train for 12 epochs on the RetrievalBench corpus with a learning rate of 3e-4 and batch size 256.
Warmup covers the first 500 steps. All runs use 8 A100 GPUs.

## 3 Datasets

RetrievalBench contains 48,000 documents averaging 9,400 tokens each. The held-out split has
4,800 documents. We also evaluate on LongQA, which has 12,000 questions.

## 4 Results

Table 1 reports accuracy on the held-out split.

Table 1: Accuracy on RetrievalBench held-out split.

| Model | Params | Accuracy | Latency (ms) |
| --- | --- | --- | --- |
| Dense baseline | 7.0B | 61.2 | 340 |
| SparseRoute k=2 | 7.0B | 70.1 | 180 |
| SparseRoute k=4 | 7.0B | 74.8 | 210 |
| SparseRoute k=8 | 7.0B | 75.1 | 295 |

SparseRoute at k = 4 improves accuracy by 13.6 points over the dense baseline while cutting
latency by 38 percent. Moving from k = 4 to k = 8 adds only 0.3 accuracy for 85 ms of latency,
so we keep k = 4 as the default described in Section 2.1.

## 5 Ablations

Removing the load-balancing term drops accuracy to 68.4 and collapses 21 of 32 experts. Raising
the routing temperature to 1.5 drops accuracy to 71.0. Training for 6 epochs instead of 12 gives
72.3 accuracy, so the extra epochs are worth 2.5 points.

## References

[1] Chen et al. Dense Long-Context Baselines. 2024.

[2] Rao et al. Expert Routing at Scale. 2025.
