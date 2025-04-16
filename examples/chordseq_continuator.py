import re
from pathlib import Path

from ctor.variable_order_markov import Variable_order_Markov


if __name__ == '__main__':
    content = Path("../data/chord_sequences.txt").read_text()
    content = content.replace("\n", ";")
    chord_labels = [label.strip() for label in content.split(";")]
    vo = Variable_order_Markov(chord_labels, None, 3)
    length = 10
    seq = vo.sample_sequence(length, constraints={0: vo.get_viewpoint('C'), length-1: vo.get_viewpoint('C')})
    result = ' '.join(seq)
    result = re.sub(r"\s([?.!,:;”])", r"\1", result)
    print(result)
