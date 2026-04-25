from lectrace import text, note, plot
import numpy as np

def main():
    text("# Backpropagation")

    text("""
        We know gradient descent needs $\\nabla_{\\mathbf{w}} L$ to update each weight.
        But a network can have millions of parameters — how do we compute all those
        gradients efficiently?

        **Backpropagation** is the algorithm that does it.
        It applies the chain rule layer by layer, flowing gradients backward from
        the loss to every parameter in one pass.
    """)

    text("## The chain rule")

    text("""
        If $L = f(g(x))$, the chain rule gives us:

        $$\\frac{dL}{dx} = \\frac{dL}{df} \\cdot \\frac{df}{dg} \\cdot \\frac{dg}{dx}$$

        Backpropagation applies this rule across every layer of the network,
        reusing intermediate results so no computation is wasted.
        That's why it's efficient — each gradient is computed exactly once.
    """)

    text("## Loss function: cross-entropy")

    text("""
        For classification with $C$ classes, we use cross-entropy loss:

        $$L = -\\log(p_y)$$

        where $p_y$ is the predicted probability of the true class $y$.

        - Correct prediction with high confidence → $L \\approx 0$
        - Wrong prediction with high confidence → $L$ is large
        - Uniform prediction (uncertain) → $L = \\log C$
    """)

    probs = np.array([0.05, 0.02, 0.88, 0.03, 0.02])  # 5-class example  # @inspect probs
    y_true = 2                                          # @inspect y_true

    loss = cross_entropy(probs, y_true)   # @inspect loss

    text("## A minimal two-layer network")

    text("""
        Let's trace the full forward pass then backpropagate by hand.
        This is the same math that PyTorch and JAX do automatically —
        understanding it once is worth more than years of black-box usage.
    """)

    np.random.seed(42)
    x  = np.array([0.6, -0.4, 1.2])     # input, 3 features  # @inspect x
    W1 = np.random.randn(3, 4) * 0.5    # @inspect W1
    b1 = np.zeros(4)                     # @inspect b1
    W2 = np.random.randn(4, 2) * 0.5    # @inspect W2
    b2 = np.zeros(2)                     # @inspect b2

    y_true_2 = 1   # class 1 is correct  # @inspect y_true_2

    text("## Forward pass")

    z1 = x @ W1 + b1      # @inspect z1
    a1 = relu(z1)          # @inspect a1
    z2 = a1 @ W2 + b2     # @inspect z2
    probs2 = softmax(z2)   # @inspect probs2
    loss2 = cross_entropy(probs2, y_true_2)  # @inspect loss2

    text(f"Forward pass — predicted class {int(np.argmax(probs2))}, loss = **{loss2:.4f}**")

    text("## Backward pass — output layer")

    text("""
        The gradient of cross-entropy + softmax w.r.t. the logits $z_2$ is beautifully simple:

        $$\\frac{\\partial L}{\\partial z_2} = p - \\mathbf{1}_y$$

        Subtract 1 from the predicted probability of the true class. That's it.
        All the complexity of softmax + log cancels out to this clean expression.
    """)

    dL_dz2 = probs2.copy()    # @inspect dL_dz2
    dL_dz2[y_true_2] -= 1

    dL_dW2 = np.outer(a1, dL_dz2)   # @inspect dL_dW2
    dL_db2 = dL_dz2.copy()           # @inspect dL_db2

    text("## Backward pass — hidden layer")

    text("""
        The gradient flows back through W2, then through the ReLU gate
        (which zeroes out gradients for negative pre-activations),
        then splits to give us gradients for W1 and b1.

        The ReLU gradient is a binary mask: 1 where `z1 > 0`, 0 elsewhere.
        Neurons that were inactive during the forward pass receive no gradient signal.
    """)

    dL_da1 = dL_dz2 @ W2.T           # @inspect dL_da1
    dL_dz1 = dL_da1 * relu_grad(z1)  # @inspect dL_dz1

    dL_dW1 = np.outer(x, dL_dz1)    # @inspect dL_dW1
    dL_db1 = dL_dz1.copy()           # @inspect dL_db1

    text("## One gradient descent step")

    lr = 0.1   # @inspect lr

    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2

    z1_new   = x @ W1 + b1
    a1_new   = relu(z1_new)
    z2_new   = a1_new @ W2 + b2
    probs_new = softmax(z2_new)
    loss_new  = cross_entropy(probs_new, y_true_2)   # @inspect loss_new

    text(f"After one update — loss: **{loss2:.4f}** → **{loss_new:.4f}** ✓")

    note("""
        In practice, autograd (PyTorch, JAX, TensorFlow) builds a computation graph
        during the forward pass and runs backprop automatically when you call .backward().
        But the math is exactly what we just did by hand.
    """)

    text("## Mini-batch training loop")

    text("""
        Real training repeats backpropagation over many random **mini-batches** —
        small subsets of the dataset typically of size 32–512.

        This is **Stochastic Gradient Descent (SGD)**: the gradient of a mini-batch
        is a noisy but unbiased estimate of the full gradient. The noise actually
        helps — it prevents the optimizer from getting stuck in sharp minima.

        One full pass over all mini-batches is called an **epoch**.
    """)

    np.random.seed(0)
    N, D = 200, 3
    X_data = np.random.randn(N, D)
    y_data = (X_data[:, 0] + X_data[:, 1] > 0).astype(int)

    W1 = np.random.randn(D, 8) * 0.3
    b1 = np.zeros(8)
    W2 = np.random.randn(8, 2) * 0.3
    b2 = np.zeros(2)

    loss_curve = _train(X_data, y_data, W1, b1, W2, b2)  # @stepover
    final_epoch_loss = loss_curve[-1]   # @inspect final_epoch_loss

    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Training loss over 40 epochs",
        "width": 420,
        "height": 240,
        "data": {"values": [{"epoch": i, "loss": l} for i, l in enumerate(loss_curve)]},
        "mark": {"type": "line", "color": "#5b8"},
        "encoding": {
            "x": {"field": "epoch", "type": "quantitative", "title": "Epoch"},
            "y": {"field": "loss", "type": "quantitative", "title": "Cross-entropy loss"},
        },
    })

    text("## Why backprop is efficient")

    text("""
        A naive approach would compute each gradient separately using finite differences:
        perturb parameter $i$ by $\\epsilon$, measure how loss changes, repeat for every parameter.
        That's $O(P)$ forward passes for $P$ parameters.

        Backprop computes **all** gradients in a single backward pass: $O(1)$ passes.

        For GPT-3 with 175 billion parameters, naive finite differences would require
        175 billion forward passes per update step. Backprop does it in one.
        That's the difference between training being possible and not possible.
    """)


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def cross_entropy(probs, y_true):
    return -np.log(probs[y_true] + 1e-9)


def _train(X_data, y_data, W1, b1, W2, b2, epochs=40, lr=0.05, batch_size=32):
    N = len(X_data)
    loss_curve = []
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            batch = idx[start:start + batch_size]
            xb, yb = X_data[batch], y_data[batch]
            z1b = xb @ W1 + b1
            a1b = relu(z1b)
            z2b = a1b @ W2 + b2
            pb  = np.array([softmax(row) for row in z2b])
            epoch_loss += np.mean([-np.log(pb[i, yb[i]] + 1e-9) for i in range(len(yb))])
            dz2 = pb.copy()
            for i, yi in enumerate(yb):
                dz2[i, yi] -= 1
            dz2 /= len(yb)
            dW2 = a1b.T @ dz2; db2 = dz2.sum(0)
            da1 = dz2 @ W2.T
            dz1 = da1 * relu_grad(z1b)
            dW1 = xb.T @ dz1; db1 = dz1.sum(0)
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
        loss_curve.append(float(epoch_loss))
    return loss_curve


if __name__ == "__main__":
    main()
