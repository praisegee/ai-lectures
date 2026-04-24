from lectrace import text, note, plot
import numpy as np

def main():
    text("# Linear Algebra for AI")

    text("""
        Data is numbers arranged in structure. Linear algebra is the mathematics
        of that structure — vectors, matrices, and the operations that transform them.

        You cannot read a machine learning paper without this vocabulary.
        This lecture gives you exactly what you need and nothing more.
    """)

    text("## Vectors")

    text("""
        A vector is an ordered list of numbers. In ML, each data point is a vector
        where every element is one *feature* — a measurable property of the thing we're modeling.
    """)

    user = np.array([28, 1, 3, 72000])  # age, is_student, years_experience, salary  # @inspect user
    item = np.array([4.5, 120, 1])      # rating, num_reviews, in_stock             # @inspect item

    text("""
        The *dimensionality* of a vector is its length — the number of features.
        `user` lives in $\\mathbb{R}^4$, `item` in $\\mathbb{R}^3$.

        Most real datasets are high-dimensional: images are $224 \\times 224 \\times 3 = 150{,}528$ dimensional,
        text tokens are embedded in spaces with thousands of dimensions.
    """)

    text("## Dot product")

    text("""
        The dot product $\\mathbf{a} \\cdot \\mathbf{b} = \\sum_i a_i b_i$ measures how much
        two vectors point in the same direction.

        It is the **core operation of every neural network layer** — a weighted sum
        of inputs, repeated once per neuron.
    """)

    weights = np.array([0.1, 0.5, 0.3, 0.0])   # @inspect weights
    score = np.dot(user, weights)                # @inspect score

    text(f"The weighted score for this user is **{score:.2f}**.")

    text("## Matrices")

    text("""
        A matrix is a 2D array of numbers. A dataset of $N$ examples each with $D$ features
        is an $N \\times D$ matrix — one row per example, one column per feature.

        Matrices let us process entire batches of data with a single operation.
    """)

    X = np.array([
        [28, 1,  3,  72000],
        [34, 0,  8,  95000],
        [22, 1,  1,  41000],
        [45, 0, 15, 130000],
    ])  # @inspect X

    text(f"Shape: {X.shape[0]} examples × {X.shape[1]} features")

    text("## Matrix–vector multiplication")

    text("""
        Multiplying $X\\mathbf{w}$ applies the same linear transformation to every row simultaneously.
        This is how a neural network layer processes an entire batch in one GPU kernel call —
        all examples in parallel, all neurons in parallel.
    """)

    w = np.array([0.1, 0.5, 0.3, 0.0])  # @inspect w
    scores = X @ w                        # @inspect scores

    text("Each element of `scores` is the dot product of one row with `w`.")

    text("## Norms — measuring vector size")

    text("""
        The L2 norm $\\|\\mathbf{v}\\|_2 = \\sqrt{\\sum_i v_i^2}$ is the Euclidean length of a vector.

        Normalizing a vector divides by its norm, giving a **unit vector** that encodes
        direction only, with magnitude exactly 1.
    """)

    v = np.array([3.0, 4.0])             # @inspect v
    norm = np.linalg.norm(v)             # @inspect norm
    v_unit = v / norm                    # @inspect v_unit

    text("## Cosine similarity")

    text("""
        Two vectors are similar if they point in roughly the same direction.

        $$\\text{cosine similarity} = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|}$$

        This captures similarity regardless of magnitude — a short and a long vector
        can be perfectly similar if they point the same way.
    """)

    doc_a = np.array([1.0, 0.8, 0.0, 0.3])   # word frequencies  # @inspect doc_a
    doc_b = np.array([0.9, 0.7, 0.1, 0.4])   # @inspect doc_b
    doc_c = np.array([0.0, 0.1, 1.0, 0.9])   # @inspect doc_c

    sim_ab = np.dot(doc_a, doc_b) / (np.linalg.norm(doc_a) * np.linalg.norm(doc_b))  # @inspect sim_ab
    sim_ac = np.dot(doc_a, doc_c) / (np.linalg.norm(doc_a) * np.linalg.norm(doc_c))  # @inspect sim_ac

    text(f"""
        - `doc_a` vs `doc_b`: **{sim_ab:.3f}** — very similar topic
        - `doc_a` vs `doc_c`: **{sim_ac:.3f}** — different topic

        This is exactly how embedding search works: store every document as a vector,
        then find the nearest neighbor by cosine similarity at query time.
    """)

    note("This is the core of semantic search, RAG pipelines, and recommendation systems.")

    text("## Eigenvalues and PCA preview")

    text("""
        A matrix $A$ has eigenvector $\\mathbf{v}$ if $A\\mathbf{v} = \\lambda\\mathbf{v}$.
        The scalar $\\lambda$ is the eigenvalue — how much that direction is stretched or compressed.

        **PCA** (Principal Component Analysis) finds the eigenvectors of the covariance matrix
        to identify the directions of maximum variance in the data.
        Drop the directions with small eigenvalues to compress dimensions with minimal information loss.
    """)

    cov = np.cov(X.T)          # @inspect cov
    eigenvalues, eigenvectors = np.linalg.eig(cov)  # @stepover
    eigenvalues = eigenvalues.real   # @inspect eigenvalues

    text("""
        The largest eigenvalue corresponds to the direction of maximum variance — the first principal component.
        In this dataset, the salary feature dominates because its scale is much larger than the others.
    """)

    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Explained variance by principal component",
        "width": 350,
        "height": 220,
        "data": {"values": [
            {"pc": f"PC{i+1}", "variance": float(abs(ev))}
            for i, ev in enumerate(sorted(abs(eigenvalues), reverse=True))
        ]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "pc", "type": "ordinal", "title": "Principal Component"},
            "y": {"field": "variance", "type": "quantitative", "title": "Eigenvalue (variance explained)"},
        },
    })
