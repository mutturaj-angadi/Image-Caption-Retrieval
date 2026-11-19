| Model | T2I R@1 | T2I R@5 | T2I R@10 | T2I mean rank | I2T R@1 | I2T R@10 | Train time (s) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| Linear regression baseline | 39.0 | 78.8 | 88.8 | 4.4 | 53.0 | 94.0 | 3.2 | 0.221 |
| GRU (random init) | pending | pending | pending | pending | pending | pending | pending | pending |
| GRU + GloVe | 17.2 | 55.8 | 74.6 | 9.2 | 18.0 | 69.0 | 2.3 | 0.81 |
| LSTM + GloVe | 21.6 | 55.0 | 72.2 | 8.9 | 23.0 | 74.0 | 2.3 | 0.599 |
