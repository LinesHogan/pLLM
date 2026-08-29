"""Negative fixture: ConfigKind must agree with the declared default type."""

from tllm.workload import ConfigField, ConfigKind, WorkloadConfig, config_field


direct_mismatch = ConfigField(True, ConfigKind.FLOAT)


class InvalidConfigKinds(WorkloadConfig):
    bool_as_float: bool = config_field(default=True, kind=ConfigKind.FLOAT)
    int_as_bool: int = config_field(default=1, kind=ConfigKind.BOOL)
    float_as_int: float = config_field(default=1.0, kind=ConfigKind.INT)
    str_as_tuple: str = config_field(default="x", kind=ConfigKind.INT_TUPLE)
    tuple_as_str: tuple[int, ...] = config_field(
        default=(1,), kind=ConfigKind.STR
    )
