# GTTA-Inspired Consumer

This consumer is a tLLM-native rewrite of GTTA's environment-feedback idea. It
accepts trajectory feedback records and maintains local action scores scoped by
task and state.

It does not import WebArena, BFCL, browser automation frameworks, or external
agent evaluation stacks.

## Supported Core

`GTTAConsumer` supports:

- bounded trajectory feedback enqueueing,
- numeric reward parsing,
- local action-score updates keyed by `task_id`, state signature, and action,
- candidate action ranking for a matching task/state,
- export of retained observation and dynamics text,
- query of retained dynamics records by task, state, and action,
- disabled no-op processing.

The first version expects callers to provide environment observations and
rewards. It does not run the environment itself.

## Trajectory Records

`TrajectoryFeedback` accepts the legacy positional shape:

```python
TrajectoryFeedback("task", "click", "observation", 1.0)
```

It also accepts the GTTA DG-style transition shape:

```python
TrajectoryFeedback(
    task_id="task",
    action="click [147] where [147] is Go",
    previous_observation="OpenStreetMap page with search results",
    next_observation="A detailed map view started loading",
    reward=1.0,
    environmental_dynamics="The map loaded a city-centered result view.",
)
```

When `previous_observation` is omitted, the legacy `observation` field is used
as the previous state text.

## Scoring and Ranking

During processing, the consumer updates both:

- a compatibility global action score, used by `score_action(action)`, and
- a state-aware action score keyed by `(task_id, state_signature, action)`.

`state_signature` is a normalized text signature derived from the previous
observation. This keeps the same action from sharing reward across unrelated
tasks or states.

Use state-aware scoring with:

```python
consumer.score_action(
    "click [147] where [147] is Go",
    task_id="map_task",
    observation="OpenStreetMap page with search results",
)
```

Use candidate ranking with:

```python
consumer.rank_candidate_actions(
    task_id="map_task",
    observation="OpenStreetMap page with search results",
    candidate_actions=["click search", "click go", "go back"],
)
```

The ranking API returns `(action, score)` pairs sorted from highest to lowest
score. Unknown actions score `0.0`, and tied scores preserve candidate order.

## Dynamics Store, Export, And Query

`export_dynamics_records()` returns processed records as dictionaries containing:

- `task_id`
- `state_signature`
- `action`
- `previous_observation`
- `next_observation`
- `environmental_dynamics`
- `reward`

`query_dynamics_records()` returns copied records filtered by any combination of:

- `task_id`
- `action`
- `observation`, normalized with `state_signature()`
- `state_signature`, when the caller already has a normalized signature

This is intentionally a local data export only; the consumer does not import
WebArena, BFCL, browsergym, or browser automation frameworks.

## Counters

Use `read_stats()` to inspect:

- `queued_trajectories`
- `dropped_trajectories`
- `processed_trajectories`
- `update_count`

An enabled smoke run should report `processed_trajectories > 0` and
`update_count > 0`.
