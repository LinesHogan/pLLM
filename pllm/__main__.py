"""Allow `python -m pllm ...` as an alias for the `pllm` console script."""

from pllm.cli import main

raise SystemExit(main())
