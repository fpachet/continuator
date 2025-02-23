import time

strings = []

t0 = time.perf_counter()
for i in range(10_000):
    for j in range(1000):
        strings.append("a" * (i // (1+j)))
print(time.perf_counter() - t0)