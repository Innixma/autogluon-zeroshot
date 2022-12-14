from contextlib import contextmanager
from time import perf_counter


@contextmanager
def catchtime(name: str) -> float:
    start = perf_counter()
    try:
        print(f"start: {name}")
        yield lambda: perf_counter() - start
    finally:
        print(f"Time for {name}: {perf_counter() - start:.4f} secs")
