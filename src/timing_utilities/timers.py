import time
from types import FunctionType
from functools import wraps
from logs_generator.logging_script import get_log_object_named

log_ = get_log_object_named('timer')


def time_fn(fn: FunctionType):
    @wraps(fn)
    def measure_time(*args, **kwargs) -> FunctionType:
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        rounded_time = round(t2-t1, 2)
        log_.info('@time_fn:' + fn.__name__ + ' took ' + str(rounded_time) + ' seconds')
        return result
    return measure_time
