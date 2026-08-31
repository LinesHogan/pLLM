"""Closed CompileOptions manifest for packed block-output resources."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from tllm.contracts.hidden_locator import HiddenSite
from tllm.schedule import (
    CompileOptions,
    HiddenLayerKind,
    PackedHiddenResourceBinding,
)


def test_packed_hidden_resource_binding_is_a_closed_typed_locator() -> None:
    indexed = PackedHiddenResourceBinding(
        layer="-1",
        layer_kind=HiddenLayerKind.INDEX,
        site=HiddenSite.BLOCK_OUTPUT,
        module_path="model.layers.3",
    )
    path = PackedHiddenResourceBinding(
        layer="target",
        layer_kind=HiddenLayerKind.PATH,
        site=HiddenSite.BLOCK_OUTPUT,
        module_path="model.layers.7",
    )

    assert indexed.locator_key == ("-1", "index", "block_output")
    assert path.locator_key == ("target", "path", "block_output")

    for changes, message in (
        ({"layer": "00"}, "canonical signed 64-bit integer"),
        ({"layer": str(1 << 63)}, "canonical signed 64-bit integer"),
        ({"layer_kind": HiddenLayerKind.WILDCARD}, "index or path"),
        ({"layer_kind": HiddenLayerKind.NONE}, "index or path"),
        ({"site": HiddenSite.POST}, "block_output"),
        ({"module_path": ""}, "must be non-empty"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(indexed, **changes)


def test_compile_options_v2_wire_requires_explicit_sorted_unique_manifest() -> None:
    bindings = (
        PackedHiddenResourceBinding(
            layer="-1",
            layer_kind=HiddenLayerKind.INDEX,
            site=HiddenSite.BLOCK_OUTPUT,
            module_path="model.layers.-1",
        ),
        PackedHiddenResourceBinding(
            layer="source",
            layer_kind=HiddenLayerKind.PATH,
            site=HiddenSite.BLOCK_OUTPUT,
            module_path="model.layers.0",
        ),
    )
    options = CompileOptions(packed_hidden_resource_manifest=bindings)
    encoded = json.loads(options.to_json())

    assert options.schema_version == 2
    assert encoded["schema_version"] == 2
    assert encoded["packed_hidden_resource_manifest"] == [
        {
            "layer": "-1",
            "layer_kind": "index",
            "module_path": "model.layers.-1",
            "site": "block_output",
        },
        {
            "layer": "source",
            "layer_kind": "path",
            "module_path": "model.layers.0",
            "site": "block_output",
        },
    ]
    assert CompileOptions.from_json(options.to_json()) == options

    missing = dict(encoded)
    del missing["packed_hidden_resource_manifest"]
    with pytest.raises(ValueError, match="schema v2 requires an explicit"):
        CompileOptions.from_json(json.dumps(missing))

    with pytest.raises(ValueError, match="sorted unique locators"):
        replace(options, packed_hidden_resource_manifest=tuple(reversed(bindings)))
    with pytest.raises(ValueError, match="sorted unique locators"):
        replace(options, packed_hidden_resource_manifest=(bindings[0], bindings[0]))


def test_compile_options_v1_accepts_only_an_empty_packed_manifest() -> None:
    legacy = CompileOptions.from_json(
        json.dumps(
            {
                "memory_budget_bytes": 64,
                "allow_serial_fallback": True,
                "schema_version": 1,
            }
        )
    )
    explicit_empty = CompileOptions.from_json(
        json.dumps(
            {
                "memory_budget_bytes": 64,
                "allow_serial_fallback": True,
                "packed_hidden_resource_manifest": [],
                "schema_version": 1,
            }
        )
    )

    assert legacy == explicit_empty
    assert legacy.packed_hidden_resource_manifest == ()
    assert json.loads(legacy.to_json())["packed_hidden_resource_manifest"] == []

    with pytest.raises(ValueError, match="schema v1 requires an empty"):
        replace(
            legacy,
            packed_hidden_resource_manifest=(
                PackedHiddenResourceBinding(
                    layer="0",
                    layer_kind=HiddenLayerKind.INDEX,
                    site=HiddenSite.BLOCK_OUTPUT,
                    module_path="model.layers.0",
                ),
            ),
        )
