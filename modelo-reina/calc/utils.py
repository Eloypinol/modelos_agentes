import os
import json
import importlib
import hashlib
from functools import wraps
from calc.perf import PerfCounter
from calc.variables import get_variable

_dataset_cache = {}


def calcfunc(variables=None, datasets=None, funcs=None, filedeps=None):
    if datasets is not None:
        assert isinstance(datasets, (list, tuple, dict))
        if not isinstance(datasets, dict):
            datasets = {x: x for x in datasets}

    if variables is not None:
        assert isinstance(variables, (list, tuple, dict))
        if not isinstance(variables, dict):
            variables = {x: x for x in variables}

    if funcs is not None:
        assert isinstance(funcs, (list, tuple))
        for func in funcs:
            assert callable(func) or isinstance(func, str)

    if filedeps is not None:
        assert isinstance(filedeps, (list, tuple))
        for filedep in filedeps:
            assert isinstance(filedep, str)
            assert os.path.getmtime(filedep)

    def wrapper_factory(func):
        func.variables = variables
        func.datasets = datasets
        func.calcfuncs = funcs
        func.filedeps = filedeps

        @wraps(func)
        def wrap_calc_func(*args, **kwargs):
            only_if_in_cache = kwargs.pop('only_if_in_cache', False)
            skip_cache = kwargs.pop('skip_cache', False)
            var_store = kwargs.pop('variable_store', None)

            assert 'variables' not in kwargs
            assert 'datasets' not in kwargs

            unknown_kwargs = set(kwargs.keys()) - set(['step_callback'])
            if not args and not unknown_kwargs and not skip_cache:
                should_cache_func = True
            else:
                should_cache_func = False

            if variables is not None:
                kwargs['variables'] = {x: get_variable(y, var_store=var_store) for x, y in variables.items()}

            ret = func(*args, **kwargs)

            return ret

        return wrap_calc_func

    return wrapper_factory
