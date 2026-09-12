from . import ops as ops
from contextlib import AbstractContextManager

def enable_torch_wrap(enable: bool = True) -> AbstractContextManager[None]: ...
