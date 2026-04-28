from lectrace import text, note, link, plot
import numpy as np


def main():
    text("# The Transformer")
    text("""
        In 2017, Vaswani et al. replaced recurrence entirely with attention.
        The result, the Transformer, became the foundation for every large
        language model in use today: GPT, Llama, Gemini, Claude.

        We'll build a minimal transformer block from scratch using numpy.
        No magic, just matrix multiplications and the right structure.
    """)

    link(
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        authors=["Vaswani", "Shazeer", "Parmar", "Uszkoreit"],
        date="2017",
        description="Introduced the Transformer. Every modern LLM descends from this paper.",
    )

    text("## Token Embeddings")
    text("""
        A transformer takes a sequence of tokens and represents each as a
        vector of dimension $d_{\\text{model}}$.

        Here: 4 tokens, each embedded in 8 dimensions.
    """)
    np.random.seed(42)
    seq_len, d_model = 4, 8
    tokens = ["the", "cat", "sat", "down"]  # @inspect tokens
    X = np.random.randn(seq_len, d_model)   # @inspect X

    text(f"Shape: `{X.shape}` ({seq_len} tokens × {d_model} dimensions)")

    text("## Queries, Keys, and Values")
    text("""
        Self-attention projects each token into three roles:

        - **Query** (Q): "what am I looking for?"
        - **Key** (K): "what do I contain?"
        - **Value** (V): "what do I contribute?"

        Each is a learned linear projection.
    """)
    d_k = 8
    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_k)

    Q = X @ W_Q  # @inspect Q
    K = X @ W_K  # @inspect K
    V = X @ W_V  # @inspect V

    text("## Attention Scores")
    text("""
        Score every token pair: how much should token $i$ attend to token $j$?

        $$\\text{scores} = \\frac{QK^{\\top}}{\\sqrt{d_k}}$$

        Dividing by $\\sqrt{d_k}$ prevents dot products from growing too large,
        which would push softmax into flat regions with tiny gradients.
    """)
    scores = Q @ K.T / np.sqrt(d_k)  # @inspect scores

    text("## Softmax: Scores to Weights")
    text("""
        Softmax normalises each row into a probability distribution:

        $$\\text{attn}_{ij} = \\frac{e^{\\text{score}_{ij}}}{\\sum_k e^{\\text{score}_{ik}}}$$

        Each row sums to 1, representing how much token $i$ attends to each position.
    """)
    attn = softmax(scores)  # @inspect attn

    note("Row sums are all 1.0, so each token distributes attention across all positions.")

    text("## Context Vectors")
    text("""
        Multiply attention weights by $V$ to get a **context vector** for each token,
        a weighted blend of all values where the weights encode relevance.

        $$\\text{output} = \\text{attn} \\cdot V$$
    """)
    context = attn @ V  # @inspect context

    text(f"Output shape: `{context.shape}`, same as input, ready for the next layer.")

    text("## Multi-Head Attention")
    text("""
        Running attention once gives one view of the sequence.
        **Multi-head attention** runs $h$ heads in parallel, each with its own
        projections, then concatenates and projects back to $d_{\\text{model}}$.

        Different heads specialise: one may track syntax, another coreference,
        another positional relationships.
    """)
    n_heads = 2
    multi_out = multi_head_attention(X, n_heads=n_heads, d_model=d_model)  # @inspect multi_out

    text(f"Multi-head output: `{multi_out.shape}` ({n_heads} heads merged back to {d_model}d).")

    text("## Feed-Forward Layer")
    text("""
        After attention, each token passes independently through a small MLP:

        $$\\text{FFN}(x) = \\max(0,\\, xW_1 + b_1)\\,W_2 + b_2$$

        The hidden dimension is typically $4 \\times d_{\\text{model}}$.
        This layer provides the non-linearity and most of the model's capacity.
    """)
    ff_out = feed_forward(multi_out, d_model=d_model, d_ff=d_model * 4)  # @inspect ff_out

    text("## Residual Connections + Layer Norm")
    text("""
        Two stabilisers wrap every sub-layer:

        1. **Residual connection**: $x + \\text{sublayer}(x)$, which lets gradients
           flow directly backward, enabling very deep networks.

        2. **Layer norm**: normalise each token vector to zero mean, unit variance.
           Prevents activations from exploding or vanishing during training.
    """)
    block_out = layer_norm(multi_out + ff_out)  # @inspect block_out

    text("## The Full Transformer Block")
    text("""
        One transformer block: attention -> residual+norm -> FFN -> residual+norm.

        Stack $N$ identical blocks. With $N=12$, $d_{\\text{model}}=768$, $h=12$ heads:
        that's **GPT-2** (124M parameters). Scale to $N=96$, $d=12288$: GPT-3 (175B).
    """)

    note("The Transformer has no recurrence and no convolution, just pure attention and matrix multiplications. That's why it parallelises so efficiently on GPUs and TPUs.")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def multi_head_attention(X: np.ndarray, n_heads: int, d_model: int) -> np.ndarray:
    np.random.seed(0)
    head_dim = d_model // n_heads
    heads = []
    for _ in range(n_heads):
        wq = np.random.randn(d_model, head_dim) * 0.1
        wk = np.random.randn(d_model, head_dim) * 0.1
        wv = np.random.randn(d_model, head_dim) * 0.1
        q = X @ wq
        k = X @ wk
        v = X @ wv
        a = softmax(q @ k.T / np.sqrt(head_dim))
        heads.append(a @ v)
    concat = np.concatenate(heads, axis=-1)
    W_O = np.random.randn(d_model, d_model) * 0.1
    return concat @ W_O


def feed_forward(X: np.ndarray, d_model: int, d_ff: int) -> np.ndarray:
    np.random.seed(1)
    W1 = np.random.randn(d_model, d_ff) * 0.1
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.1
    b2 = np.zeros(d_model)
    hidden = np.maximum(0, X @ W1 + b1)
    return hidden @ W2 + b2


def layer_norm(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)


if __name__ == "__main__":
    main()
