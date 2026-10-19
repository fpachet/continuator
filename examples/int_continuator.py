"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

from ctor.sequence_model import SequenceModel

if __name__ == '__main__':
    train_seq = [1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10]
    vo = SequenceModel(kmax=10)
    vo.learn_sequence(train_seq)
    # zeroseq = vo.sample_zero_order(21)
    # result = ' '.join(str(i) for i in zeroseq)
    # result = re.sub(r"\s([?.!,:;”])", r"\1", result)
    # print("zero order integer sequence:")
    # print(result)
    seq = vo.sample_sequence(22, constraints={0: vo.start_symbol, 10: 6, 21: vo.end_symbol})
    if not seq:
        print("no solution")
    else:
        print("constrained integer sequence:")
        print(seq[1:-1])
