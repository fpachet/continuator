"""
Copyright (c) 2025 Ynosound.
All rights reserved.

See LICENSE file in the project root for full license information.
"""
from pathlib import Path
import re

from ctor.variable_order_markov import Variable_order_Markov

if __name__ == '__main__':
    data_path = Path(__file__).resolve().parents[1] / "data" / "proust_debut.txt"
    with open(data_path, 'r') as file:
        recherche = file.read().rstrip()
    char_seq = list(recherche)
    train_seq = re.findall(r"\w+|[^\w\s]", recherche, re.UNICODE)
    vo = Variable_order_Markov(train_seq, None, 3)
    seq = vo.sample_sequence(100, constraints={0: vo.get_viewpoint('.'), 99: vo.get_viewpoint('.')})
    result = ' '.join(seq)
    # Removes spaces before punctuation
    result = re.sub(r"\s([?.!,:;”])", r"\1", result)
    print(result)
