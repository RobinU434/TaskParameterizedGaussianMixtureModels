from functools import wraps
import time


def timeit(func):
    """Decorator that measures the execution time of the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function that measures the execution time.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper
