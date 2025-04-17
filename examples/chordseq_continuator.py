import re

from ctor.variable_order_markov import Variable_order_Markov

if __name__ == '__main__':
    # computes chord sequences of length 8 starting and ending with, say, C and with a F#7 in the middle
    with open('../data/chord_sequences.txt', 'r') as file:
        seqs = file.readlines()
    seqs = [seq.split(';')[1:-1] for seq in seqs]
    seqs = [[chord.strip() for chord in seq] for seq in seqs]
    vo = Variable_order_Markov(None, None, kmax=3)
    for seq in seqs:
        vo.learn_sequence(seq)

    length = 8
    for i in range(20):
        seq = vo.sample_sequence(length, constraints={0: vo.get_viewpoint('C'), int(length/2): vo.get_viewpoint('F#7'), length - 1: vo.get_viewpoint('C')})
        result = ' '.join(seq)
        print(result)
