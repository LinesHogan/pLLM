"""ESamp through the public execution path, eager and with step capture."""

from pllm.benchmark import BenchmarkOptions, compare
from pllm.workloads import ESampRequestConfig, ESampRequestWorkload


def main() -> None:
    compare(
        ESampRequestWorkload(ESampRequestConfig(rank=4)),
        BenchmarkOptions.parse(),
        training_rows=1,
        capture=True,
        require_training=True,
    )


if __name__ == "__main__":
    main()
