from lectrace import text, note, plot


def main():
    text("# What is Artificial Intelligence?")

    text(
        """
        AI is the science of building systems that can perform tasks that normally
        require human intelligence — perceiving, reasoning, learning, deciding.

        The goal is not to replicate human biology, but to replicate human-level
        *outcomes* on well-defined tasks: classifying images, translating text,
        playing chess, generating code.
    """
    )

    text("## The three learning paradigms")

    text(
        """
        Every ML algorithm fits into one of three categories depending on
        what signal the model learns from.
    """
    )

    paradigms = [
        {
            "category": "Supervised",
            "signal": "labeled examples",
            "example": "spam detection",
        },
        {
            "category": "Unsupervised",
            "signal": "raw data only",
            "example": "customer clustering",
        },
        {
            "category": "Reinforcement",
            "signal": "rewards & penalties",
            "example": "game playing",
        },
    ]  # @inspect paradigms

    text("## The ML pipeline")

    text(
        """
        Every project follows the same loop regardless of how simple or complex the model is.
        Understanding this loop is more useful than memorizing any particular algorithm.
    """
    )

    pipeline = [
        "collect data",
        "preprocess",
        "define model",
        "train",
        "evaluate",
        "deploy",
    ]  # @inspect pipeline

    text("## Why now?")

    text(
        """
        Three ingredients converged to produce the current wave of AI:

        - **Data** — internet-scale datasets with billions of labeled examples
        - **Compute** — GPUs that parallelize matrix operations at enormous scale
        - **Algorithms** — deep learning, attention, and self-supervised pretraining

        Remove any one and the AI revolution stalls. All three arrived together in the 2010s.
    """
    )

    text("## What we'll build in this series")

    steps = {
        "lecture 02": "linear algebra — the language of data",
        "lecture 03": "gradient descent — how models learn",
        "lecture 04": "neural networks — universal function approximators",
        "lecture 05": "backpropagation — computing gradients at scale",
        "lecture 06": "attention — the engine behind modern AI",
    }  # @inspect steps

    note(
        """
        Ask the audience: what's an example of AI they used today?
        Common answers: search autocomplete, spam filter, face unlock, recommendation feeds.
        Point out that all of them run on the same core ideas we're about to cover.
    """
    )

    text("## A simple prediction problem")

    text(
        """
        Suppose we observe that studying more hours leads to higher exam scores.
        We want a function $f$ such that:

        $$f(\\text{hours}) \\approx \\text{score}$$

        This is the essence of supervised learning — find a function that maps
        inputs to outputs by learning from examples.
    """
    )

    hours = [1, 2, 3, 4, 5, 6, 7, 8]  # @inspect hours
    scores = [45, 52, 61, 70, 75, 82, 88, 95]  # @inspect scores

    plot(
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "Hours studied vs exam score",
            "width": 400,
            "height": 250,
            "data": {
                "values": [{"hours": h, "score": s} for h, s in zip(hours, scores)]
            },
            "mark": {"type": "point", "size": 80, "filled": True, "color": "#4f8ef7"},
            "encoding": {
                "x": {
                    "field": "hours",
                    "type": "quantitative",
                    "title": "Hours studied",
                },
                "y": {
                    "field": "score",
                    "type": "quantitative",
                    "title": "Exam score",
                    "scale": {"zero": False},
                },
            },
        }
    )

    result = compute_sum(3, 9)  # @inspect result

    text(
        """
        A **model** is a parameterized function. **Training** finds the parameters
        that make the function fit the data. Everything else in this course is details
        about how to do that efficiently and correctly.
    """
    )


def compute_sum(a, b):  # @inspect
    total = a + b
    return total


if __name__ == "__main__":
    main()
