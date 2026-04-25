from lectrace import text, note, plot
import numpy as np


def main():
    text("# Language Models")

    text("""
        A language model assigns a probability to a sequence of tokens.
        Equivalently, it predicts the next token given everything before:

        $$P(x_1, x_2, \\ldots, x_n) = \\prod_{t=1}^{n} P(x_t \\mid x_1, \\ldots, x_{t-1})$$

        This is the foundation of every LLM — GPT, Llama, Gemini, Claude.
        They all do one thing: predict the next token, one at a time.
    """)

    text("## Tokenisation")

    text("""
        Before modelling, text is broken into **tokens** — subword units in
        real models, single characters here for simplicity.
    """)

    corpus = "the cat sat on the mat the cat ate the rat"  # @inspect corpus
    chars = sorted(set(corpus))                             # @inspect chars
    vocab_size = len(chars)                                 # @inspect vocab_size

    stoi = {c: i for i, c in enumerate(chars)}  # @inspect stoi
    itos = {i: c for c, i in stoi.items()}

    tokens = [stoi[c] for c in corpus]  # @inspect tokens

    text(f"Vocabulary: {vocab_size} characters. Corpus: {len(tokens)} tokens.")

    text("## Bigram Model — Count Co-occurrences")

    text("""
        The simplest language model: count how often each character follows every other.

        $$P(x_t = c \\mid x_{t-1} = p) = \\frac{\\text{count}(p \\to c)}{\\sum_{c'} \\text{count}(p \\to c')}$$

        This is a $|V| \\times |V|$ table of transition probabilities.
    """)

    counts = count_bigrams(tokens, vocab_size)  # @inspect counts @stepover
    probs = counts / counts.sum(axis=1, keepdims=True)  # @inspect probs

    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Bigram transition probabilities",
        "width": 320,
        "height": 320,
        "data": {
            "values": [
                {"from": chars[i], "to": chars[j], "prob": float(probs[i, j])}
                for i in range(vocab_size)
                for j in range(vocab_size)
            ]
        },
        "mark": "rect",
        "encoding": {
            "x": {"field": "to",   "type": "nominal", "title": "Next character"},
            "y": {"field": "from", "type": "nominal", "title": "Current character"},
            "color": {"field": "prob", "type": "quantitative", "scale": {"scheme": "blues"}},
            "tooltip": [
                {"field": "from"},
                {"field": "to"},
                {"field": "prob", "format": ".3f"},
            ],
        },
    })

    text("## Generating Text — Sampling")

    text("""
        To generate text, we sample the next token from the model's distribution
        and feed it back in as context — **autoregressive generation**.
    """)

    generated = generate(probs, itos, stoi, seed_char="t", n=40)  # @inspect generated @stepover

    text(f"Generated: `{generated}`")

    text("## Temperature — Controlling Randomness")

    text("""
        **Temperature** $T$ reshapes the probability distribution before sampling:

        $$P_T(c) = \\frac{P(c)^{1/T}}{\\sum_{c'} P(c')^{1/T}}$$

        - $T \\to 0$: deterministic — always pick the most likely token
        - $T = 1$: sample from the learned distribution as-is
        - $T > 1$: flatten the distribution — more random and creative
    """)

    low_T  = generate(probs, itos, stoi, seed_char="t", n=40, temperature=0.1)   # @inspect low_T @stepover
    high_T = generate(probs, itos, stoi, seed_char="t", n=40, temperature=2.5)  # @inspect high_T @stepover

    text(f"T=0.1 (focused): `{low_T}`")
    text(f"T=2.5 (chaotic): `{high_T}`")

    note("Most production LLMs default to T ≈ 0.7–1.0. Lower T for factual tasks, higher for creative ones.")

    text("## Perplexity — Measuring Model Quality")

    text("""
        **Perplexity** measures how surprised the model is by the data.
        Lower is better — a perfect model has perplexity 1, a uniform random
        model has perplexity $|V|$.

        $$\\text{PPL} = \\exp\\!\\left(-\\frac{1}{N} \\sum_{t=1}^{N} \\log P(x_t \\mid x_{t-1})\\right)$$
    """)

    ppl = perplexity(probs, tokens)  # @inspect ppl @stepover

    text(f"Bigram perplexity: **{ppl:.2f}** (vs {vocab_size} for a random model)")

    text("## From Bigrams to Transformers")

    text("""
        The bigram model looks back exactly 1 token. Real LLMs look back thousands:

        | Model | Context | Params |
        |-------|---------|--------|
        | Bigram | 1 token | vocab² |
        | GPT-2 | 1,024 tokens | 124M |
        | GPT-3 | 2,048 tokens | 175B |
        | GPT-4 | 128,000 tokens | ~1T |

        The core objective — predict $P(x_t \\mid x_{<t})$ — never changes.
        The Transformer just makes it work at massive context and scale.
    """)

    note("Every word in a ChatGPT response came from sampling this distribution, one token at a time.")


def count_bigrams(tokens: list[int], vocab_size: int) -> np.ndarray:
    counts = np.ones((vocab_size, vocab_size), dtype=np.float32)  # +1 smoothing
    for a, b in zip(tokens[:-1], tokens[1:]):
        counts[a, b] += 1
    return counts


def generate(
    probs: np.ndarray,
    itos: dict,
    stoi: dict,
    seed_char: str,
    n: int,
    temperature: float = 1.0,
) -> str:
    np.random.seed(0)
    current = stoi[seed_char]
    out = [seed_char]
    for _ in range(n - 1):
        row = probs[current].astype(np.float64)
        if temperature != 1.0:
            row = row ** (1.0 / temperature)
        row = np.clip(row, 0, None)
        row /= row.sum()  # always renormalise to avoid float precision drift
        current = int(np.random.choice(len(row), p=row))
        out.append(itos[current])
    return "".join(out)


def perplexity(probs: np.ndarray, tokens: list[int]) -> float:
    log_sum = 0.0
    for a, b in zip(tokens[:-1], tokens[1:]):
        p = float(probs[a, b])
        log_sum += np.log(max(p, 1e-10))
    return float(np.exp(-log_sum / (len(tokens) - 1)))


if __name__ == "__main__":
    main()
