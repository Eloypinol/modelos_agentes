import os
from .utils import calcfunc


class ExecutionInterrupted(Exception):
    pass


def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


__all__ = [calcfunc]