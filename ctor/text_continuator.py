from ctor.variable_order_markov import Variable_order_Markov


def non_overlapping_tuples(lst, k):
    return [tuple(lst[i:i+k]) for i in range(0, len(lst) - k + 1, k)]

if __name__ == '__main__':
    # with open('../../data/proust_debut.txt', 'r') as file:
    #     recherche = file.read().rstrip()
    # train_seq = list(recherche)
    # train_seq =  re.findall(r"\w+|[^\w\s]", recherche, re.UNICODE)
    # train_seq =  non_overlapping_tuples(train_seq, 2)
    train_seq = [1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10]
    vo = Variable_order_Markov(train_seq, None, 10)

# zero order
#     zeroseq = vo.sample_zero_order(20)
#     result = ' '.join(zeroseq)
#     result = re.sub(r"\s([?.!,:;”])", r"\1", result)
#     print(result)  # Removes spaces before punctuation

# variable order
#     seq = vo.sample_sequence(vo.get_viewpoint('.'), 10, vo.get_viewpoint('.'))
#     result = ' '.join(seq)
#     result = re.sub(r"\s([?.!,:;”])", r"\1", result)
#     print(result)# Removes spaces before punctuation

    # seq = vo.sample_sequence(vo.get_start_vp(), 12, vo.get_end_vp())
    # seq = vo.sample_sequence_that_ends(vo.get_start_vp(), 16)
    seq = vo.sample_sequence(20, constraints={0: vo.start_padding, 19: vo.end_padding})
    print(seq)