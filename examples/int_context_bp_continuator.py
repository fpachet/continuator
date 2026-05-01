"""
Integer sequence example using the experimental context-BP core.
"""

from ctor.core import ContextBPModel


if __name__ == "__main__":
    train_seq = [1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10]
    model = ContextBPModel(kmax=10, seed=0)
    model.learn_sequence(train_seq)

    graph, _, _ = model.infer(21, constraints={9: 6, 20: model.end_symbol})
    seq = model.sample_sequence(21, constraints={9: 6, 20: model.end_symbol})

    if not seq:
        print("no solution")
    else:
        print(f"constrained integer sequence, effective order {graph.kmax}:")
        print(seq[:-1])
