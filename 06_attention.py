from lectrace import text, note, plot
import numpy as np

def main():
    text("# Attention and Transformers")
    text("""
        Attention is the mechanism that lets a model decide which parts of the input
        to focus on when producing each output.

        It is the core innovation of the **Transformer**, the architecture behind
        GPT, BERT, Claude, Gemini, and every major language model today.
        Before attention, models struggled with long-range dependencies.
        Attention solved that completely.
    """)

    text("## The problem attention solves")
    text("""
        In a sequence, not every word is equally relevant to every other word.

        > *"The cat sat on the mat because **it** was comfortable."*

        Resolving "it" requires attending back to "cat", not "mat" or "sat".
        Earlier architectures (RNNs, LSTMs) had to compress the entire past into
        a fixed-size vector, losing information over long distances.

        Attention lets every token directly inspect every other token in $O(1)$ steps,
        regardless of distance.
    """)

    text("## Queries, Keys, and Values")
    text("""
        Each token produces three learned projections:

        - **Query** $Q$: "what am I looking for?"
        - **Key** $K$: "what do I offer to other queries?"
        - **Value** $V$: "what information do I actually contain?"

        The attention score between token $i$ and token $j$ is $Q_i \\cdot K_j$.
        A high score means $i$ should attend strongly to $j$.
    """)

    text("## Scaled dot-product attention")
    text("""
        $$\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$

        The scaling by $\\sqrt{d_k}$ is critical. Without it, dot products grow large
        in high dimensions, pushing softmax into saturation where gradients vanish.
    """)

    text("## Step-by-step: three tokens")
    text("""
        Let's trace attention on a tiny sequence: `[the, cat, sat]`
        Each token is embedded into a 4-dimensional vector.
    """)
    np.random.seed(3)
    d_model = 4
    tokens = ["the", "cat", "sat"]  # @inspect tokens

    embeddings = np.array([
        [ 0.2,  0.8,  0.1, -0.3],
        [ 0.9,  0.4, -0.1,  0.7],
        [ 0.3, -0.2,  0.6,  0.5],
    ])  # @inspect embeddings

    W_Q = np.random.randn(d_model, d_model) * 0.5   # @inspect W_Q
    W_K = np.random.randn(d_model, d_model) * 0.5   # @inspect W_K
    W_V = np.random.randn(d_model, d_model) * 0.5   # @inspect W_V

    Q = embeddings @ W_Q   # @inspect Q
    K = embeddings @ W_K   # @inspect K
    V = embeddings @ W_V   # @inspect V

    text("## Computing attention scores")
    d_k = d_model
    raw_scores = Q @ K.T / np.sqrt(d_k)   # @inspect raw_scores

    text("""
        Each row corresponds to one token attending to all others.
        Higher score means more attention. Negative means less relevant.
    """)
    attention_weights = softmax(raw_scores)   # @inspect attention_weights

    text("""
        After softmax, each row sums to 1, giving a probability distribution over which
        tokens to attend to. This is what gets visualized in attention heatmaps.
    """)
    output = attention_weights @ V   # @inspect output

    text("""
        The output for each token is a weighted average of all Value vectors.
        Tokens that scored high contribute more to the result.
        The network learns to weight information from relevant tokens automatically.
    """)

    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Attention weights (row=query, col=key)",
        "width": 220,
        "height": 220,
        "data": {"values": [
            {"query": tokens[i], "key": tokens[j], "weight": float(attention_weights[i, j])}
            for i in range(3) for j in range(3)
        ]},
        "mark": "rect",
        "encoding": {
            "x": {"field": "key", "type": "ordinal", "title": "Key (attended to)"},
            "y": {"field": "query", "type": "ordinal", "title": "Query (attending from)"},
            "color": {"field": "weight", "type": "quantitative", "title": "Weight",
                      "scale": {"scheme": "blues"}},
        },
    })

    text("## Multi-head attention")
    text("""
        One attention head captures one type of relationship,
        such as syntactic structure, coreference, or topic similarity.

        **Multi-head attention** runs $H$ heads in parallel, each with its own $W_Q, W_K, W_V$,
        then concatenates and projects the results:

        $$\\text{MultiHead}(Q,K,V) = \\text{concat}(\\text{head}_1, \\ldots, \\text{head}_H)\\, W^O$$

        Different heads learn to look for different relationship types simultaneously.
    """)
    n_heads = 2
    d_head = d_model // n_heads   # @inspect d_head

    heads = []
    for h in range(n_heads):   # @stepover
        Wq = np.random.randn(d_model, d_head) * 0.3
        Wk = np.random.randn(d_model, d_head) * 0.3
        Wv = np.random.randn(d_model, d_head) * 0.3
        Qh, Kh, Vh = embeddings @ Wq, embeddings @ Wk, embeddings @ Wv
        out_h, _ = scaled_dot_product_attention(Qh, Kh, Vh)
        heads.append(out_h)

    multi_head_out = np.concatenate(heads, axis=-1)   # @inspect multi_head_out

    note("GPT-3: 96 attention heads, each operating in 128D. 175 billion parameters total.")

    text("## The full Transformer block")
    text("""
        A Transformer layer wraps multi-head attention with three additions that
        make training stable and deep stacking possible:

        1. **Residual connection**: $x \\leftarrow x + \\text{Attention}(x)$,
           gradients flow directly through the skip connection, preventing vanishing gradients

        2. **Layer normalization**: applied after each sub-layer,
           stabilizes the distribution of activations during training

        3. **Feed-forward network**: two linear layers applied to each token independently,
           adds per-token capacity beyond what attention provides

        Stack 12 of these blocks to get BERT. Stack 96 to get GPT-3.
    """)

    text("## Why Transformers won")
    reasons = {
        "parallelism": "all tokens attend simultaneously, no sequential bottleneck like RNNs",
        "long range":  "any token attends to any other in O(1) steps regardless of distance",
        "scale":       "more compute + more data = reliably better performance",
        "pretraining": "train once on internet text, fine-tune for any downstream task",
    }  # @inspect reasons

    text("## From attention to modern AI")
    text("""
        GPT models are **decoder-only Transformers** trained on one objective:
        predict the next token. Given enough text and compute, this simple task
        forces the model to internalize grammar, facts, reasoning, and code.

        The attention mechanism lets it pull context from thousands of tokens back,
        reaching across an entire document to resolve ambiguity, maintain consistency,
        and build coherent arguments.

        Everything in modern AI that isn't a neural network backbone uses the attention
        mechanism: image generation (DiT), protein folding (AlphaFold), code synthesis (Copilot),
        speech recognition (Whisper). The architecture you just traced through is the
        foundation of the current AI moment.
    """)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights


if __name__ == "__main__":
    main()
