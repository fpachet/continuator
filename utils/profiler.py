import time
from functools import wraps
from typing import Any


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.6f} seconds")
        return result

    return timeit_wrapper


class Timeit(object):
    all_func: dict[Any, dict[str, float | int]] = {}

    def __init__(self, func):
        self.call_count = 0
        self.total_time = 0.0
        self.decorated_instance = None
        self.func = func
        Timeit.all_func[func] = {
            "call_count": self.call_count,
            "total_time": self.total_time,
        }

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        t0 = time.perf_counter()
        res = self.func(self.decorated_instance, *args, **kwargs)
        self.total_time += time.perf_counter() - t0
        Timeit.all_func[self.func]["call_count"] += 1
        Timeit.all_func[self.func]["total_time"] = self.total_time
        return res

    def __get__(self, obj, objtype):
        self.decorated_instance = obj
        return self

    def __str__(self):
        return f"{self.call_count}"

    def info(self):
        print(
            f"Function {self.func.__name__}:"
            f"\n\t- total time: {self.total_time:.6f}s"
            f"\n\t- avg. time: {(self.total_time / self.call_count):.6f}s "
            f"({self.call_count} calls)"
        )

    @classmethod
    def all_info(cls):
        for func in cls.all_func:
            t = cls.all_func[func]["total_time"]
            n = cls.all_func[func]["call_count"]
            if n:
                avg = t / n
                print(f"{func.__name__}: {t:.6f}s / {avg:.6f}s / {n}")
            else:
                ...
                # print(f"{func.__name__}: no calls")
