from lectrace import text, note, plot
import numpy as np

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def predict(X, w, b):
    return X * w + b

def _train(X, y_true, w, b, lr=0.001, steps=200):
    history = []
    for _ in range(steps):
        y_pred = predict(X, w, b)
        error = y_pred - y_true
        w -= lr * 2 * np.mean(error * X)
        b -= lr * 2 * np.mean(error)
        history.append(float(mse_loss(y_pred, y_true)))
    return w, b, history

def _compare_lrs(X, y_true, learning_rates, steps=200):
    results = {}
    for lr_val in learning_rates:
        np.random.seed(0)
        w, b = np.random.randn(), np.random.randn()
        history = []
        for _ in range(steps):
            y_p = predict(X, w, b)
            e = y_p - y_true
            w -= lr_val * 2 * np.mean(e * X)
            b -= lr_val * 2 * np.mean(e)
            history.append(float(mse_loss(y_p, y_true)))
        results[lr_val] = history
    return results

def main():
    text("# Gradient Descent")

    text("""
        Training a model means finding parameters that minimize a loss function.
        Gradient descent is the algorithm that does it — follow the slope of the
        loss surface downhill, one step at a time.

        It is arguably the most important algorithm in all of machine learning.
        Every model you've heard of — GPT, Stable Diffusion, AlphaFold — was
        trained with some variant of gradient descent.
    """)

    text("## The loss function")

    text("""
        A loss function measures how wrong the model's predictions are.
        For regression, we use **Mean Squared Error**:

        $$L(\\mathbf{w}) = \\frac{1}{N}\\sum_{i=1}^{N}(\\hat{y}_i - y_i)^2$$

        where $\\hat{y}_i$ is the prediction and $y_i$ is the true value.
        Perfect predictions → $L = 0$. Worse predictions → larger $L$.
    """)

    y_true = np.array([45, 52, 61, 70, 75, 82, 88, 95], dtype=float)  # @inspect y_true

    text("## Starting with random weights")

    text("""
        Before training, we initialize `w` and `b` randomly.
        The model makes terrible predictions — that's expected.
        Training will fix it.
    """)

    np.random.seed(0)
    w = np.random.randn()   # @inspect w
    b = np.random.randn()   # @inspect b

    X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)  # @inspect X

    y_pred = predict(X, w, b)        # @inspect y_pred
    loss = mse_loss(y_pred, y_true)  # @inspect loss

    text(f"Initial loss: **{loss:.2f}** — random weights predict poorly.")

    text("## The gradient")

    text("""
        The gradient $\\nabla L$ is a vector of partial derivatives — one per parameter —
        pointing in the direction of **steepest increase** of the loss.

        We want the loss to *decrease*, so we step in the **opposite** direction.

        For MSE linear regression, the gradients have a closed form:

        $$\\frac{\\partial L}{\\partial w} = \\frac{2}{N}\\sum_i (\\hat{y}_i - y_i)\\, x_i$$

        $$\\frac{\\partial L}{\\partial b} = \\frac{2}{N}\\sum_i (\\hat{y}_i - y_i)$$
    """)

    text("## The update rule")

    text("""
        $$w \\leftarrow w - \\eta\\,\\frac{\\partial L}{\\partial w}$$

        where $\\eta$ (eta) is the **learning rate** — a small positive number controlling
        how large each step is. Too large and we overshoot. Too small and training is slow.
    """)

    text("## Training loop — watch the loss fall")

    w, b, loss_history = _train(X, y_true, w, b)  # @stepover
    w = float(w)    # @inspect w
    b = float(b)    # @inspect b
    final_loss = loss_history[-1]  # @inspect final_loss

    text(f"After 200 steps — loss: **{final_loss:.2f}**, w: **{w:.3f}**, b: **{b:.3f}**")

    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Loss over training steps",
        "width": 420,
        "height": 240,
        "data": {"values": [{"step": i, "loss": l} for i, l in enumerate(loss_history)]},
        "mark": {"type": "line", "color": "#e45"},
        "encoding": {
            "x": {"field": "step", "type": "quantitative", "title": "Step"},
            "y": {"field": "loss", "type": "quantitative", "title": "MSE Loss"},
        },
    })

    text("## The learning rate matters enormously")

    text("""
        The learning rate is the most important hyperparameter in training.

        - **Too large** — steps overshoot the minimum, loss oscillates or diverges
        - **Too small** — converges correctly but wastes compute
        - **Just right** — smooth descent to a good minimum

        In practice: start at `1e-3`, halve it if training becomes unstable.
    """)

    results = _compare_lrs(X, y_true, [0.1, 0.01, 0.001, 0.0001])  # @stepover

    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Effect of learning rate on convergence",
        "width": 420,
        "height": 240,
        "data": {"values": [
            {"step": i, "loss": min(l, 5000), "lr": str(lr_val)}
            for lr_val, history in results.items()
            for i, l in enumerate(history)
        ]},
        "mark": "line",
        "encoding": {
            "x": {"field": "step", "type": "quantitative", "title": "Step"},
            "y": {"field": "loss", "type": "quantitative", "title": "Loss (capped at 5000)"},
            "color": {"field": "lr", "type": "nominal", "title": "Learning rate"},
        },
    })

    note("A common heuristic: start at 1e-3, reduce by 10× if training is unstable.")

    text("## What we learned")

    takeaways = [
        "loss functions quantify how wrong the model is",
        "gradients point toward steepest increase in loss",
        "we step opposite the gradient, scaled by learning rate",
        "iterate until convergence — that's training",
    ]  # @inspect takeaways

    text("""
        Everything in modern ML training — Adam, momentum, learning rate schedules,
        gradient clipping — is an enhancement on top of this basic loop.
        The core idea never changes.
    """)
