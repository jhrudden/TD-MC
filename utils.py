import pickle
from typing import Any, Callable

def load_or_generate_data(filepath: str, callable_: Callable, *args, extractor: Callable = lambda x: x, **kwargs) -> Any:
    """
    Load the data from filepath if it exists, otherwise call callable and extract the data using extractor, then pickle the data and return it.

    Args:
        filepath: The filepath to check for the data
        callable_: The callable to call if the data doesn't exist
        *args: The arguments to pass to callable_
        extractor: The callable to extract the data from callable_ (defaults to identity function)
        *kwargs: The keyword arguments to pass to callable
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = extractor(callable_(*args, **kwargs))
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    return data