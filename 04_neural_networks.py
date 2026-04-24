from lectrace import text, note, plot
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

class Linear:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features)

    def __call__(self, x):
        return x @ self.W + self.b

    def __lectrace__(self):
        return {"W_shape": list(self.W.shape), "b_shape": list(self.b.shape)}

def main():
    text("# Neural Networks")

    text("""
        A neural network is a composition of simple functions —
        linear transformations interleaved with nonlinearities.

        Stacked deep enough, they can approximate *any* continuous function
        to arbitrary precision. This is the **Universal Approximation Theorem**,
        and it's why neural networks work for such a wide range of tasks.
    """)

    text("## The neuron")

    text("""
        A single neuron computes:

        $$z = \\mathbf{w} \\cdot \\mathbf{x} + b, \\qquad a = \\sigma(z)$$

        - $\\mathbf{w}$: weights — learned parameters
        - $b$: bias — learned offset
        - $\\sigma$: activation function — introduces nonlinearity
        - $z$: pre-activation (weighted sum)
        - $a$: activation (output of the neuron)
    """)

    x = np.array([0.5, -1.2, 0.8, 2.1])   # one input example, 4 features  # @inspect x
    w = np.array([0.3, -0.7, 1.1, 0.2])   # @inspect w
    b = 0.5                                 # @inspect b

    z = np.dot(x, w) + b   # @inspect z
    a = relu(z)             # @inspect a

    text(f"Pre-activation z = **{z:.3f}**, after ReLU a = **{a:.3f}**")

    text("## Activation functions")

    text("""
        Without activation functions, every layer is just a linear transform.
        Composing linear functions always gives another linear function — no matter
        how many layers you stack, the whole network collapses to a single matrix.

        Activations break this linearity and give the network its expressive power.
        The three most common:

        - **ReLU** $= \\max(0, z)$ — simple, fast, works well in most cases
        - **Sigmoid** $= \\frac{1}{1 + e^{-z}}$ — outputs in (0, 1), good for probabilities
        - **Tanh** $= \\frac{e^z - e^{-z}}{e^z + e^{-z}}$ — outputs in (-1, 1), zero-centered
    """)

    z_range = np.linspace(-4, 4, 80)  # @stepover

    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Activation functions",
        "width": 420,
        "height": 240,
        "data": {"values": [
            {"z": float(z), "a": float(a), "fn": name}
            for name, values in {
                "ReLU": relu(z_range),
                "Sigmoid": sigmoid(z_range),
                "Tanh": np.tanh(z_range),
            }.items()
            for z, a in zip(z_range, values)
        ]},
        "mark": "line",
        "encoding": {
            "x": {"field": "z", "type": "quantitative", "title": "z (pre-activation)"},
            "y": {"field": "a", "type": "quantitative", "title": "activation"},
            "color": {"field": "fn", "type": "nominal"},
        },
    })

    text("## Layers")

    text("""
        A *layer* applies a linear transform to all inputs simultaneously — one set of weights per output neuron.
        With $D_{\\text{in}}$ input features and $D_{\\text{out}}$ neurons, the weight matrix is $D_{\\text{in}} \\times D_{\\text{out}}$.

        In code this is just a matrix multiply: `output = input @ W + b`
    """)

    layer = Linear(4, 3, seed=42)   # @inspect layer
    out = layer(x)                   # @inspect out

    text("## Building a deep network")

    text("""
        A network is layers chained together. The output of one becomes the input of the next.
        We'll build a 3-layer network for classifying handwritten digits (10 classes).

        **Architecture: 784 → 256 → 128 → 10**

        - **Input**: 784 = 28×28 flattened pixel values
        - **Hidden layer 1**: 256 neurons, ReLU activation
        - **Hidden layer 2**: 128 neurons, ReLU activation
        - **Output**: 10 neurons, one per digit class (0–9)
    """)

    l1 = Linear(784, 256, seed=0)   # @inspect l1
    l2 = Linear(256, 128, seed=1)   # @inspect l2
    l3 = Linear(128, 10,  seed=2)   # @inspect l3

    text("## Forward pass")

    text("""
        A *forward pass* feeds an input through the network layer by layer to produce a prediction.
        Each layer transforms its input with a linear operation followed by a nonlinearity.
        The final layer produces raw scores called *logits*, which softmax converts to probabilities.
    """)

    np.random.seed(7)
    image = np.random.rand(784)   # simulated flattened 28×28 image  # @inspect image

    h1 = relu(l1(image))   # @inspect h1
    h2 = relu(l2(h1))      # @inspect h2
    logits = l3(h2)        # @inspect logits
    probs = softmax(logits)  # @inspect probs

    predicted_digit = int(np.argmax(probs))    # @inspect predicted_digit
    confidence = float(probs[predicted_digit]) # @inspect confidence

    text(f"Predicted digit: **{predicted_digit}** with confidence **{confidence:.1%}**")

    note("""
        With random weights the prediction is meaningless — confidence means nothing yet.
        Training adjusts all the weights so the correct digit gets high probability.
    """)

    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Output probabilities (untrained network)",
        "width": 380,
        "height": 200,
        "data": {"values": [{"digit": str(i), "prob": float(p)} for i, p in enumerate(probs)]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "digit", "type": "ordinal", "title": "Digit"},
            "y": {"field": "prob", "type": "quantitative", "title": "Probability", "axis": {"format": ".0%"}},
        },
    })

    text("## Counting parameters")

    text("""
        Each layer has $D_{\\text{in}} \\times D_{\\text{out}}$ weights plus $D_{\\text{out}}$ biases.
        Our tiny digit classifier already has over 200,000 parameters.
        GPT-3 has 175 billion.
    """)

    params = {
        "layer 1 (784→256)": 784 * 256 + 256,
        "layer 2 (256→128)": 256 * 128 + 128,
        "layer 3 (128→10)":  128 * 10  + 10,
    }  # @inspect params

    total = sum(params.values())   # @inspect total
    text(f"Total trainable parameters: **{total:,}**")

    text("## Hierarchical feature learning")

    text("""
        What makes deep networks powerful is not the individual neurons —
        it's that layers build on each other to learn increasingly abstract representations.

        For a vision network trained on images:
        - **Layer 1** detects edges, corners, and color gradients
        - **Layer 2** combines edges into textures and simple shapes
        - **Layer 3** combines shapes into object parts (wheels, eyes, ears)
        - **Deeper layers** combine parts into objects

        Nobody designs these features — the network discovers them automatically
        through training. This emergent hierarchy is what makes deep learning so powerful.
    """)
