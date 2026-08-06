# Revisiting Expert Routing Under Latency Budgets

## 1 Setup

We re-evaluate SparseRoute on RetrievalBench under a fixed 200 ms latency budget. Unlike the
original report, we measure end-to-end latency including the router itself.

## 2 Findings

Table 1: Re-measured accuracy under a 200 ms budget.

| Model | Accuracy | Latency (ms) |
| --- | --- | --- |
| Dense baseline | 61.2 | 340 |
| SparseRoute k=4 | 72.9 | 244 |

SparseRoute k=4 reaches 72.9 accuracy in our setup rather than the 74.8 originally reported, and
its measured latency of 244 ms exceeds the 200 ms budget. The gap comes from router overhead that
the original latency figures excluded.

## 3 Recommendation

Under a strict latency budget, SparseRoute k=2 is the better operating point on RetrievalBench.
