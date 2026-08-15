# greedy decode 对比（HF vs ref/model.py）

- prompt: `Explain the concept of a transformer neural network and its attention mechanism:`
- 新 token 数: 20
- **逐 token 一致: ✅ 是**

| step | HF | ref |
|---|---|---|
| 0 | 3555 | 3555 |
| 1 | 374 | 374 |
| 2 | 279 | 279 |
| 3 | 6672 | 6672 |
| 4 | 1948 | 1948 |
| 5 | 264 | 264 |
| 6 | 42578 | 42578 |
| 7 | 323 | 323 |
| 8 | 264 | 264 |
| 9 | 29728 | 29728 |
| 10 | 3922 | 3922 |
| 11 | 304 | 304 |
| 12 | 4586 | 4586 |
| 13 | 30 | 30 |
| 14 | 3555 | 3555 |
| 15 | 374 | 374 |
| 16 | 279 | 279 |
| 17 | 3476 | 3476 |
| 18 | 315 | 315 |
| 19 | 279 | 279 |

HF 解码文本: ` What is the difference between a transformer and a neural network in general? What is the role of the`
