# AI Lectures

An interactive, step-through lecture series covering the foundations of modern AI —
from basic linear algebra to the Transformer architecture behind GPT, Claude, and Gemini.

Built with [lectrace](https://github.com/praisegee/lectrace): write Python, get an
interactive viewer on GitHub Pages automatically.

---

## Lectures

| # | Title | Topics |
|---|-------|--------|
| 01 | **What is AI?** | Learning paradigms, the ML pipeline, prediction problems |
| 02 | **Linear Algebra** | Vectors, dot products, matrix multiplication, cosine similarity, PCA |
| 03 | **Gradient Descent** | Loss functions, gradients, the update rule, learning rate effects |
| 04 | **Neural Networks** | Neurons, activation functions, layers, forward pass, parameter counting |
| 05 | **Backpropagation** | Chain rule, computing gradients by hand, mini-batch SGD |
| 06 | **Attention & Transformers** | Q/K/V, scaled dot-product attention, multi-head attention, the Transformer block |

Each lecture is a self-contained Python file. Step through execution with arrow keys,
inspect live variable values, and see interactive charts — all in the browser.

---

## Run locally

```sh
pip install lectrace numpy
lectrace serve
```

Opens at `http://localhost:7000`. File changes trigger an automatic rebuild.

---

## How it works

Each lecture is a plain Python file that imports from `lectrace`:

```python
from lectrace import text, plot, note
import numpy as np

def main():
    text("# Gradient Descent")
    text("""
        Training a model means finding parameters that minimize a loss function.
        Gradient descent is how we do it — follow the slope downhill, step by step.
    """)

    w = 0.5   # @inspect w
    loss = (w - 2.0) ** 2  # @inspect loss

    note("Ask: what happens if the learning rate is too large?")
```

- `# @inspect x` — shows the variable in the side panel after that line runs
- `text("""...""")` — renders Markdown with LaTeX math support
- `plot({...})` — embeds an interactive Vega-Lite chart
- `note("...")` — adds a presenter annotation (toggle with `N`)

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `→` / `j` | Next step |
| `←` / `k` | Previous step |
| `Shift+→` | Step over (skip into function calls) |
| `Shift+←` | Step back over |
| `↑` | Step up (out of current function) |
| `N` | Toggle presenter notes |
| `E` | Toggle variable panel |

---

## Deploy to GitHub Pages

```sh
lectrace init   # generates .github/workflows/lectrace.yml
git add .
git commit -m "add lectures"
git push
```

Enable **Settings → Pages → Source: GitHub Actions** in your repo.
Every push to `main` rebuilds and redeploys automatically.
