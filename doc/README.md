# tLLM Documentation

tLLM is a test-time intervention layer for vLLM: declare your algorithm as a small dataflow program, and the compiler derives fast execution from the declarations.

## Reading order

1. [Quickstart](quickstart.md) — install, run the starter, attach a consumer.
2. [Write a consumer](write-a-consumer.md) — the program model: `@flow`, `@extern`, state banks, and what the compiler derives.
3. [Architecture](architecture.md) — the layered stack, the import gate, and how programs are compiled and executed.
4. [ESamp reference](esamp.md) — the built-in test-time training consumer.

## The one-paragraph mental model

You write a `torch.nn.Module` and annotate its methods with `@flow(inputs/outputs/state)` and `@extern(inputs/after/state/deadline)`. `tllm.program.compile_program` slices each flow into torch.export graphs, lowers the optimizer into functional transitions, and verifies the compiled transition against your Python body at compile time. The generated consumer runs inside vLLM generation: state writes go to a training lane, read-only predictions go to a reader lane, and whole steps are captured into CUDA graphs — all derived from the dependency edges you declared.

If your program uses something the language doesn't support, compilation fails loudly with a stable error code (`PG###`). There is no silent fallback.

中文文档见 [doc_zh/README.md](../doc_zh/README.md)。
