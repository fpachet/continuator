from collections import Counter

import numpy as np

lst= np.arange(1, 1000, 1)
tuples = list(zip(*(lst[i:] for i in range(5))))
conts = zip(tuples, lst[5:])
for ctx, cont in conts:
    print(ctx, cont)
print(Counter(conts))
