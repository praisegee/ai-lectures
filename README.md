# AI Lectures

An interactive step-through lecture series covering the foundations of modern AI —
from linear algebra and gradient descent to the Transformer architecture powering
today's large language models.

---

## Lectures

| # | Title | Topics covered |
|---|-------|---------------|
| 01 | **What is AI?** | Learning paradigms, the ML pipeline, supervised vs unsupervised vs reinforcement learning |
| 02 | **Linear Algebra** | Vectors, dot products, matrix multiplication, cosine similarity, PCA |
| 03 | **Gradient Descent** | Loss functions, gradients, the update rule, learning rate effects |
| 04 | **Neural Networks** | Neurons, activation functions, layers, forward pass, parameter counting |
| 05 | **Backpropagation** | Chain rule, manual gradient computation, mini-batch SGD |
| 06 | **Attention & Transformers** | Queries/keys/values, scaled dot-product attention, multi-head attention, the Transformer block |

---

## Access

**Online** — browse and step through any lecture in your browser:
> https://praisegee.github.io/ai-lectures/

**Locally** — clone and run with a live-reload server:

```sh
git clone https://github.com/praisegee/ai-lectures.git
cd ai-lectures
pip install lectrace numpy
lectrace serve
```

Opens at `http://localhost:7000`. Use arrow keys to step through execution,
inspect live variable values, and interact with charts.

---

Lectures are prepared with [lectrace](https://github.com/praisegee/lectrace).
