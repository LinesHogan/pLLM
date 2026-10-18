"""Public ESamp eager/capture comparison."""

from pllm.benchmark import BenchmarkOptions, compare
from pllm.workloads import ESampRequestConfig, ESampRequestWorkload


def main() -> None:
    paged = ESampRequestWorkload(ESampRequestConfig(rank=4, paged=True))
    compare(
        ESampRequestWorkload(ESampRequestConfig(rank=4)),
        BenchmarkOptions.parse(),
        explicit_math=True,
        capture=True,
        require_training=True,
        alternatives=(("paged", paged, False), ("paged_capture", paged, True)),
    )


if __name__ == "__main__":
    main()
