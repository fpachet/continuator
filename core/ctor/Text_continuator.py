import random

import core
from core.ctor.variable_order_markov import Variable_order_Markov
import re


if __name__ == '__main__':
    with open('../../data/proust_du_cote.txt', 'r') as file:
        recherche = file.read().rstrip()
    # train_seq = list(recherche)
    train_seq =  re.findall(r"\w+|[^\w\s]", recherche, re.UNICODE)
    vo = Variable_order_Markov(train_seq, None, 10)

# zero order
    zeroseq = vo.sample_zero_order(20)
    result = ' '.join(zeroseq)
    result = re.sub(r"\s([?.!,:;”])", r"\1", result)
    print(result)  # Removes spaces before punctuation

# variable order
    seq = vo.sample_sequence(vo.get_viewpoint('.'), -1, vo.get_viewpoint('.'))
    result = ' '.join(seq)
    result = re.sub(r"\s([?.!,:;”])", r"\1", result)
    print(result)# Removes spaces before punctuation

