import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        print(f"Function '{func.__name__}' executed in {time_end - time_start:.4f} seconds")
        return result
    return wrapper