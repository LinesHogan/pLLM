# GTTA-Inspired Consumer

This consumer is a tLLM-native rewrite of GTTA's environment-feedback idea. It
accepts trajectory feedback records and maintains local action scores.

It does not import WebArena, BFCL, browser automation frameworks, or external
agent evaluation stacks.

## Supported Core

`GTTAConsumer` supports:

- bounded trajectory feedback enqueueing,
- numeric reward parsing,
- local action-score updates,
- disabled no-op processing.

The first version expects callers to provide environment observations and
rewards. It does not run the environment itself.

## Counters

Use `read_stats()` to inspect:

- `queued_trajectories`
- `dropped_trajectories`
- `processed_trajectories`
- `update_count`

An enabled smoke run should report `processed_trajectories > 0` and
`update_count > 0`.
