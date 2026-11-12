from collections.abc import Callable, Iterable
from types import FunctionType
from typing import Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

class JITFunction(Generic[P, R]):
    @property
    def fn(self) -> FunctionType: ...
    raw_src: list[str]
    starting_line_number: int
    version: int | None
    _repr: Callable[..., str] | None
    launch_metadata: Callable[..., object] | None
    do_not_specialize: Iterable[int | str]
    do_not_specialize_on_alignment: Iterable[int | str]
    debug: bool | None
    noinline: bool | None
    __name__: str
    def __getitem__(self, grid: tuple[int, ...]) -> Callable[P, R]: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

def jit(
    fn: Callable[P, R],
    *,
    version: int | None = None,
    repr: Callable[..., str] | None = None,
    launch_metadata: Callable[..., object] | None = None,
    do_not_specialize: Iterable[int | str] | None = None,
    do_not_specialize_on_alignment: Iterable[int | str] | None = None,
    debug: bool | None = None,
    noinline: bool | None = None,
) -> JITFunction[P, R]: ...
def next_power_of_2(n: int) -> int: ...
def cdiv(x: int, y: int) -> int: ...
