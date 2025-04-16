import re
from pathlib import Path

from ctor.variable_order_markov import Variable_order_Markov

if __name__ == '__main__':
    # computes chord sequences of length 8 starting and ending with, say, C
    with open('../data/chord_sequences.txt', 'r') as file:
        seqs = file.readlines()
    seqs = [seq.split(';')[1:-1] for seq in seqs]
    seqs = [[chord.strip() for chord in seq] for seq in seqs]
    vo = Variable_order_Markov(None, None, kmax=3)
    for seq in seqs:
        vo.learn_sequence(seq)

    for i in range(20):
        length = 8
        seq = vo.sample_sequence(length, constraints={0: vo.get_viewpoint('C'), length - 1: vo.get_viewpoint('C')})
        result = ' '.join(seq)
        result = re.sub(r"\s([?.!,:;”])", r"\1", result)
        print(result)
