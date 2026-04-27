"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""
import gzip
from pathlib import Path

from ctor.variable_order_markov import Variable_order_Markov


def open_chord_sequences():
    data_path = Path(__file__).resolve().parents[1] / "data" / "chord_sequences.txt"
    if data_path.exists():
        return open(data_path, "r", encoding="utf-8")
    return gzip.open(f"{data_path}.gz", "rt", encoding="utf-8")


if __name__ == '__main__':
    # computes chord sequences of length 8 starting and ending with, say, C and with a F#7 in the middle
    with open_chord_sequences() as file:
        seqs = file.readlines()
    seqs = [seq.split(';')[1:-1] for seq in seqs]
    seqs = [[chord.strip() for chord in seq] for seq in seqs]
    vo = Variable_order_Markov(None, None, kmax=3)
    for seq in seqs:
        vo.learn_sequence(seq)

    length = 4
    for i in range(20):
        seq = vo.sample_sequence(length, constraints={0: vo.get_viewpoint('C'), length - 1: vo.get_viewpoint('C')})
        result = ' '.join(seq)
        print(result)

    length = 8
    for i in range(20):
        seq = vo.sample_sequence(length, constraints={0: vo.get_viewpoint('C'), int(length/2): vo.get_viewpoint('F#7'), length - 1: vo.get_viewpoint('C')})
        result = ' '.join(seq)
        print(result)
