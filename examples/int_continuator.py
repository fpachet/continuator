"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""

import re

from ctor.variable_order_markov import Variable_order_Markov

if __name__ == '__main__':
    train_seq = [1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10]
    vo = Variable_order_Markov(train_seq, None, 10)
    zeroseq = vo.sample_zero_order(20)
    result = ' '.join(str(i) for i in zeroseq)
    result = re.sub(r"\s([?.!,:;”])", r"\1", result)
    print("zero order integer sequence:")
    print(result)
    seq = vo.sample_sequence(20, constraints={0: vo.start_padding, 10: 6, 19: vo.end_padding})
    print("constrained integer sequence:")
    print(seq[1:-1])
