import time
import functools

def calculate_runtime(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Function {func.__name__} took {runtime:.4f} seconds to complete.")
    return result