import re

from ctor.variable_order_markov import Variable_order_Markov

if __name__ == '__main__':
    with open('../data/proust_debut.txt', 'r') as file:
        recherche = file.read().rstrip()
    char_seq = list(recherche)
    vo = Variable_order_Markov(char_seq, None, 3)
    seq = vo.sample_sequence(140, constraints={0: vo.get_viewpoint('.'), 139: vo.get_viewpoint('.')})
    result = ''.join(seq)
    result = re.sub(r"\s([?.!,:;”])", r"\1", result)
    print(result)  # Removes spaces before punctuation
