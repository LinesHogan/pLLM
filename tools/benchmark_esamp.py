"""ESamp through the public execution path, plain and with step capture."""

import argparse

from pllm.benchmark import BenchmarkOptions, compare
from pllm.workloads import ESampConfig, ESampWorkload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=0.0, help="ESamp sampling-guidance strength")
    options, args = BenchmarkOptions.parse(parser)
    compare(
        ESampWorkload(ESampConfig(rank=4, intervention_beta=args.beta)),
        options,
        max_training_rows=1,
        require_training=True,
    )


if __name__ == "__main__":
    main()
