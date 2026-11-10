"""Extract decoder-layer fixtures by byte range, without reading or downloading full weights.

Used by architecture admission experiments, not for producing useful language models.
The source is a pinned Hub revision or a checkpoint directory on a shared filesystem.
Planning reads only metadata and safetensors headers. Quantized tensors are kept in their
stored format when the quantization config names no layer indices; conversion and MTP
are rejected until their dependency/renaming rules are implemented.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import struct
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from http.client import HTTPResponse, IncompleteRead
from pathlib import Path
from threading import Lock
from typing import BinaryIO, TypeAlias
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

JSON: TypeAlias = "None | bool | int | float | str | list[JSON] | dict[str, JSON]"
CHUNK = 8 * 1024 * 1024
LIMIT = 64 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}


def mapping(value: JSON) -> dict[str, JSON]:
    """Validate a JSON object received from a repository."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def integer(value: JSON) -> int:
    """Validate an integer received from a repository."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Expected an integer, got {value!r}")
    return value


def string(value: JSON) -> str:
    """Validate a string received from a repository."""
    if not isinstance(value, str):
        raise ValueError("Expected a string")
    return value


def sequence(value: JSON) -> list[JSON]:
    """Validate a JSON array received from a repository."""
    if not isinstance(value, list):
        raise ValueError("Expected an array")
    return value


def decode(data: bytes) -> dict[str, JSON]:
    """Read a JSON object with a typed boundary."""
    value: JSON = json.loads(data)
    return mapping(value)


def filename(name: str) -> str:
    """Reject repository paths that could escape the destination."""
    if not name or name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"Unsafe repository filename: {name!r}")
    return name


@dataclass(frozen=True)
class Tensor:
    """One selected tensor and its absolute source byte interval (end exclusive)."""

    source: str
    name: str
    shard: str
    dtype: str
    shape: list[int]
    start: int
    end: int
    file_size: int

    @property
    def size(self) -> int:
        """Number of stored tensor bytes."""
        return self.end - self.start


class Source(ABC):
    """Byte counters shared by the remote and local checkpoint readers."""

    def __init__(self) -> None:
        self._counter_lock = Lock()
        self.metadata_bytes = 0
        self.payload_bytes = 0

    @abstractmethod
    def metadata(self, name: str) -> bytes:
        """Read a small asset with a hard size cap, never a weight file; FileNotFoundError when absent."""

    @abstractmethod
    def range(self, name: str, start: int, end: int, size: int | None, sink: BinaryIO, *, payload: bool) -> int:
        """Copy bytes [start, end) of `name` to `sink` and return the file size."""

    def _record(self, count: int, payload: bool) -> None:
        with self._counter_lock:
            if payload:
                self.payload_bytes += count
            else:
                self.metadata_bytes += count


def capped(name: str, data: bytes) -> bytes:
    """Reject weight files and oversized metadata before they are parsed."""
    if name.endswith(".safetensors"):
        raise ValueError("Weight files require byte ranges")
    if len(data) > LIMIT:
        raise ValueError(f"Metadata exceeds {LIMIT} bytes: {name}")
    return data


class Local(Source):
    """Bounded reads from a checkpoint directory, e.g. a cluster's shared model store."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def metadata(self, name: str) -> bytes:
        with (self.root / filename(name)).open("rb") as stream:
            data = capped(name, stream.read(LIMIT + 1))
        self._record(len(data), False)
        return data

    def range(self, name: str, start: int, end: int, size: int | None, sink: BinaryIO, *, payload: bool) -> int:
        path = self.root / filename(name)
        total = path.stat().st_size
        if total < end or (size is not None and total != size):
            raise ValueError(f"{path} changed or is shorter than its header claims")
        with path.open("rb") as stream:
            stream.seek(start)
            offset = start
            while offset < end:
                data = stream.read(min(CHUNK, end - offset))
                if not data:
                    raise OSError(f"Short read from {path}")
                write_bytes(sink, data)
                offset += len(data)
                self._record(len(data), payload)
        return total


class Remote(Source):
    """Bounded HTTP reads with fresh Hub redirects on every retry and range chunk."""

    def __init__(self, repo: str, revision: str) -> None:
        super().__init__()
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"

    def metadata(self, name: str) -> bytes:
        request = Request(self.base + quote(filename(name)), headers={"Accept-Encoding": "identity"})
        response: HTTPResponse
        try:
            with urlopen(request, timeout=60) as response:
                data = capped(name, response.read(LIMIT + 1))
        except HTTPError as error:
            error.close()
            if error.code == 404:
                raise FileNotFoundError(name) from error
            raise
        self._record(len(data), False)
        return data

    def range(self, name: str, start: int, end: int, size: int | None, sink: BinaryIO, *, payload: bool) -> int:
        """Fetch only requested bytes; close a 200 response before reading its body."""
        offset = start
        total = size
        failures = 0
        while offset < end:
            url = self.base + quote(filename(name)) + f"?download=true&slice_request={time.time_ns()}"
            request = Request(url, headers={"Range": f"bytes={offset}-{end - 1}", "Accept-Encoding": "identity"})
            try:
                response: HTTPResponse
                with urlopen(request, timeout=90) as response:
                    total = validate_response(response, offset, end, total)
                    while offset < end:
                        data = response.read(min(CHUNK, end - offset))
                        if not data:
                            raise OSError(f"Short range response for {name}")
                        write_bytes(sink, data)
                        offset += len(data)
                        self._record(len(data), payload)
                    if response.read(1):
                        raise ValueError(f"Excess range response bytes for {name}")
            except HTTPError as error:
                error.close()
                if error.code not in (401, 403, 408, 429, 500, 502, 503, 504):
                    raise
                failures += 1
            except IncompleteRead as error:
                write_bytes(sink, error.partial)
                offset += len(error.partial)
                self._record(len(error.partial), payload)
                failures += 1
            except OSError:
                failures += 1
            if failures >= 5:
                raise OSError(f"Range retry budget exhausted for {name} at byte {offset}")
            if offset < end:
                time.sleep(2**failures)
        assert total is not None
        return total


def write_bytes(sink: BinaryIO, data: bytes) -> None:
    """Keep destination failures out of the HTTP retry path."""
    try:
        count = sink.write(data)
    except OSError as error:
        raise ValueError(f"Cannot write slice output: {error}") from error
    if count != len(data):
        raise ValueError("Short destination write; inspect disk space before retrying")


def validate_response(response: HTTPResponse, start: int, end: int, size: int | None) -> int:
    """Validate the transport before consuming any weight bytes."""
    if response.status != 206:
        raise ValueError(f"Range request returned HTTP {response.status}; aborting without reading payload")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if match is None:
        raise ValueError("Missing or malformed Content-Range")
    first, last, total = map(int, match.groups())
    if first != start or last != end - 1 or total < end or (size is not None and total != size):
        raise ValueError("Content-Range does not match the pinned file/range")
    if response.headers.get("Content-Encoding", "identity") != "identity":
        raise ValueError("Range response must use identity encoding")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != end - start:
        raise ValueError("Content-Length does not match requested range")
    return total


def read_header(remote: Source, shard: str) -> tuple[dict[str, JSON], int, int]:
    """Fetch only a safetensors length prefix and bounded JSON header."""
    buffer = io.BytesIO()
    size = remote.range(shard, 0, 8, None, buffer, payload=False)
    length = struct.unpack("<Q", buffer.getvalue())[0]
    if length < 2 or length > LIMIT or length + 8 > size:
        raise ValueError(f"Invalid safetensors header length in {shard}: {length}")
    buffer = io.BytesIO()
    remote.range(shard, 8, 8 + length, size, buffer, payload=False)
    return decode(buffer.getvalue()), 8 + length, size


def header_tensors(header: dict[str, JSON], base: int, size: int, shard: str) -> dict[str, Tensor]:
    """Validate shapes, dtype sizes, offsets and the entire shard's nonoverlap."""
    result: dict[str, Tensor] = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        entry = mapping(value)
        dtype = string(entry["dtype"])
        shape = [integer(n) for n in sequence(entry["shape"])]
        offsets = [integer(n) for n in sequence(entry["data_offsets"])]
        if len(offsets) != 2 or any(n < 0 for n in shape):
            raise ValueError(f"Invalid shape/offsets: {name}")
        start, end = offsets
        if not 0 <= start <= end <= size - base:
            raise ValueError(f"Out-of-bounds tensor: {name}")
        if dtype not in DTYPE_BYTES or math.prod(shape) * DTYPE_BYTES[dtype] != end - start:
            raise ValueError(f"Unsupported dtype or inconsistent tensor size: {name}, {dtype}")
        result[name] = Tensor(name, name, shard, dtype, shape, base + start, base + end, size)
    cursor = base
    for tensor in sorted(result.values(), key=lambda t: t.start):
        if tensor.start != cursor:
            raise ValueError(f"Overlapping or noncontiguous safetensors storage in {shard}")
        cursor = tensor.end
    if cursor != size:
        raise ValueError(f"Unindexed bytes in {shard}")
    return result


def select_layers(count: int, requested: str | None) -> list[int]:
    """Select an ordered, proper subset of source decoder layers."""
    layers = list(dict.fromkeys([0, 1, count // 2, count // 2 + 1, count - 2, count - 1]))
    if requested is not None:
        layers = [int(n) for n in requested.split(",")]
    if not layers or len(set(layers)) != len(layers) or any(n < 0 or n >= count for n in layers):
        raise ValueError("--layers must contain unique, in-range comma-separated indices")
    if len(layers) >= count:
        raise ValueError("Full-model extraction is forbidden; select fewer decoder layers")
    return layers


def qwen2_kinds(config: dict[str, JSON], layers: list[int]) -> None:
    """Qwen dense layers are uniform; sliding-window layer ranges are not remapped."""
    if config.get("use_sliding_window", False):
        raise ValueError("Sliding-window layer remapping is not implemented")
    config["max_window_layers"] = len(layers)


def qwen_moe_kinds(config: dict[str, JSON], layers: list[int]) -> None:
    """Qwen MoE: remap the dense-MLP layer list; MTP layers are dropped."""
    qwen2_kinds(config, layers)
    dense = sequence(config.get("mlp_only_layers", []))
    step = integer(config.get("decoder_sparse_step", 1))
    if step <= 0:
        raise ValueError("decoder_sparse_step must be positive")
    config["mlp_only_layers"] = [i for i, src in enumerate(layers) if src in dense or (src + 1) % step != 0]
    config["decoder_sparse_step"] = 1
    config["mtp_num_hidden_layers"] = 0


def qwen_hybrid_kinds(config: dict[str, JSON], layers: list[int]) -> None:
    """Qwen hybrids: materialize the source full/linear pattern before slicing it."""
    qwen_moe_kinds(config, layers)
    count = integer(config["num_hidden_layers"])
    interval = integer(config.get("full_attention_interval", 4))
    if interval <= 0:
        raise ValueError("full_attention_interval must be positive")
    kinds = sequence(
        config.get(
            "layer_types",
            ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)],
        )
    )
    if len(kinds) != count or any(k not in ("full_attention", "linear_attention") for k in kinds):
        raise ValueError("Unsupported hybrid layer_types; supply one full/linear kind per source layer")
    config["layer_types"] = [kinds[i] for i in layers]


def deepseek_kinds(config: dict[str, JSON], layers: list[int]) -> None:
    """DeepSeek V2/V3: the leading dense-MLP layers stay a prefix; MTP layers are dropped."""
    if integer(config.get("moe_layer_freq", 1)) != 1:
        raise ValueError("moe_layer_freq other than 1 changes layer kinds on renumbering; not implemented")
    dense = [src < integer(config["first_k_dense_replace"]) for src in layers]
    if dense != sorted(dense, reverse=True):
        raise ValueError("Select the dense first_k_dense_replace layers before MoE layers")
    config["first_k_dense_replace"] = sum(dense)
    config["num_nextn_predict_layers"] = 0


def gpt_oss_kinds(config: dict[str, JSON], layers: list[int]) -> None:
    """gpt-oss alternates sliding and full attention; keep each selected layer's kind."""
    kinds = sequence(config["layer_types"])
    if len(kinds) != integer(config["num_hidden_layers"]):
        raise ValueError("gpt-oss layer_types must list one kind per source layer")
    config["layer_types"] = [kinds[i] for i in layers]


# Per family: how a renumbered layer keeps its attention and MLP kind.
FAMILIES = {
    "Qwen2ForCausalLM": qwen2_kinds,
    "Qwen3MoeForCausalLM": qwen_moe_kinds,
    "Qwen3NextForCausalLM": qwen_hybrid_kinds,
    "Qwen3_5MoeForCausalLM": qwen_hybrid_kinds,
    "Qwen3_5ForCausalLM": qwen_hybrid_kinds,
    "DeepseekV2ForCausalLM": deepseek_kinds,
    "DeepseekV3ForCausalLM": deepseek_kinds,
    "GptOssForCausalLM": gpt_oss_kinds,
}


def rewrite_config(
    source: dict[str, JSON], layers_arg: str | None, text_only: bool
) -> tuple[dict[str, JSON], list[int]]:
    """Preserve source attention and MoE kinds while renumbering a supported family."""
    arch = sequence(source["architectures"])
    architecture = string(arch[0]) if len(arch) == 1 else ""
    config = dict(source)
    if "text_config" in source:
        if not text_only:
            raise ValueError("Multimodal checkpoints require --text-only; vision extraction is unsupported")
        config = dict(mapping(source["text_config"]))
        config["tie_word_embeddings"] = source.get("tie_word_embeddings", False)
        architecture = architecture.replace("ForConditionalGeneration", "ForCausalLM")
        if architecture == "KimiK25ForCausalLM":
            architecture = "DeepseekV3ForCausalLM"
        for key in ("quantization_config", "transformers_version"):
            if key in source:
                config[key] = source[key]
    if architecture not in FAMILIES:
        raise ValueError(f"Unsupported family {arch}; implement its dependency/config rules first")
    # The slice ships no model code; vLLM and transformers use their native classes.
    config.pop("auto_map", None)
    config["architectures"] = [architecture]
    layers = select_layers(integer(config["num_hidden_layers"]), layers_arg)
    FAMILIES[architecture](config, layers)
    config["num_hidden_layers"] = len(layers)
    if "quantization_config" in config:
        quantization = dict(mapping(config["quantization_config"]))
        if "ignore" in quantization:
            quantization["ignore"] = [text_name(string(p)) for p in sequence(quantization["ignore"])]
        config["quantization_config"] = quantization
    return config, layers


def text_name(name: str) -> str:
    """Flatten the Qwen and Kimi text prefixes, including quantization exclusions."""
    return name.replace("model.language_model.", "model.", 1).replace("language_model.", "", 1)


def selected_names(weights: dict[str, JSON], layers: list[int]) -> dict[str, str]:
    """Keep complete selected layers and the text embedding, final norm and head."""
    result: dict[str, str] = {}
    for source in weights:
        name = text_name(source)
        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
        if match is not None:
            layer = int(match[1])
            if layer in layers:
                result[source] = f"model.layers.{layers.index(layer)}.{match[2]}"
        elif name.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
            result[source] = name
    if len(set(result.values())) != len(result):
        raise ValueError("Text prefix rewriting produced duplicate tensors")
    return result


def plan(remote: Source, weights: dict[str, JSON], names: dict[str, str]) -> list[Tensor]:
    """Resolve selected tensors against validated source headers before any payload."""
    shards = sorted({filename(string(weights[name])) for name in names})
    tensors: list[Tensor] = []
    for shard in shards:
        header, base, size = read_header(remote, shard)
        entries = header_tensors(header, base, size, shard)
        for source, name in names.items():
            if weights[source] != shard:
                continue
            if source not in entries:
                raise ValueError(f"Index tensor missing from {shard}: {source}")
            t = entries[source]
            tensors.append(Tensor(source, name, shard, t.dtype, t.shape, t.start, t.end, size))
    return sorted(tensors, key=lambda t: (t.shard, t.start))


def dependency_problems(
    config: dict[str, JSON], tensors: list[Tensor], include_mtp: bool, dequantize: str | None
) -> list[str]:
    """Reject unimplemented layouts/conversions and missing decoder dependencies."""
    problems: list[str] = []
    names = {t.name for t in tensors}
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    if not config.get("tie_word_embeddings", False):
        required.add("lm_head.weight")
    problems.extend(f"Missing required tensor: {name}" for name in sorted(required - names))
    default_kinds: list[JSON] = ["full_attention"] * integer(config["num_hidden_layers"])
    kinds = sequence(config.get("layer_types", default_kinds))
    for i, kind in enumerate(kinds):
        prefix = f"model.layers.{i}."
        layer_names = [n for n in names if n.startswith(prefix)]
        attention = "linear_attn." if kind == "linear_attention" else "self_attn."
        for component in (attention, "mlp.", "input_layernorm.", "post_attention_layernorm."):
            if not any(n.startswith(prefix + component) for n in layer_names):
                problems.append(f"Layer {i} is missing {component} required by its config")
    quantization = config.get("quantization_config")
    if quantization is None:
        if any(t.dtype.startswith("F8") or "scale_inv" in t.name or "weight_packed" in t.name for t in tensors):
            problems.append("Tensors look quantized but the config has no quantization_config")
    else:
        quant = mapping(quantization)
        if quant.get("quant_method") == "compressed-tensors":
            problems.extend(int4_problems(quant, tensors))
        elif quant.get("quant_method") != "mxfp4":
            problems.append(f"Unsupported stored quantization: {quant.get('quant_method')}")
        if re.search(r"layers(?:\\\\)?\.\d", json.dumps(quantization)):
            problems.append("quantization_config names source layer indices; remapping them is not implemented")
    if dequantize is not None:
        problems.append(
            "--dequantize bf16 is a stub: no conversion formats are implemented; omit it for native BF16 weights"
        )
    if include_mtp:
        problems.append("--include-mtp is not implemented; omit it for ordinary decoding")
    return problems


def int4_problems(quant: dict[str, JSON], tensors: list[Tensor]) -> list[str]:
    """Check the symmetric group-32 INT4 packing used by Kimi K2 Thinking and K2.5–K2.7."""
    groups = mapping(quant["config_groups"])
    for value in groups.values():
        group = mapping(value)
        weight = mapping(group["weights"])
        if (
            quant.get("format") != "pack-quantized"
            or group.get("input_activations") is not None
            or group.get("output_activations") is not None
            or any(
                weight.get(k) != v
                for k, v in {
                    "num_bits": 4,
                    "group_size": 32,
                    "symmetric": True,
                    "type": "int",
                    "strategy": "group",
                    "dynamic": False,
                    "actorder": None,
                }.items()
            )
        ):
            return ["Only symmetric, static INT4 group-32 weight-only packing is supported"]
    inventory = {t.name: t for t in tensors}
    problems: list[str] = []
    for tensor in tensors:
        if not tensor.name.endswith(".weight_packed"):
            continue
        prefix = tensor.name.removesuffix("weight_packed")
        scale = inventory.get(prefix + "weight_scale")
        shape = inventory.get(prefix + "weight_shape")
        if (
            tensor.dtype != "I32"
            or len(tensor.shape) != 2
            or scale is None
            or shape is None
            or scale.dtype not in ("BF16", "F16", "F32")
            or scale.shape != [tensor.shape[0], tensor.shape[1] // 4]
            or shape.dtype not in ("I32", "I64")
            or shape.shape != [2]
        ):
            problems.append(f"Missing or inconsistent INT4 packing/scale/shape: {tensor.name}")
    return problems


def shard_groups(tensors: list[Tensor], limit: int) -> list[list[Tensor]]:
    """Bound shard payloads without splitting individual tensors."""
    groups: list[list[Tensor]] = []
    group: list[Tensor] = []
    size = 0
    for tensor in tensors:
        if group and size + tensor.size > limit:
            groups.append(group)
            group, size = [], 0
        group.append(tensor)
        size += tensor.size
    if group:
        groups.append(group)
    return groups


def write_shard(remote: Source, path: Path, tensors: list[Tensor]) -> None:
    """Serialize a new header and stream adjacent selected intervals without gaps."""
    header: dict[str, JSON] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for t in tensors:
        header[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [offset, offset + t.size]}
        offset += t.size
    data = json.dumps(header, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        i = 0
        while i < len(tensors):
            first = tensors[i]
            end = first.end
            i += 1
            while i < len(tensors) and tensors[i].shard == first.shard and tensors[i].start == end:
                end = tensors[i].end
                i += 1
            remote.range(first.shard, first.start, end, first.file_size, stream, payload=True)
    # Validate the serialized inventory without mapping or loading tensor payloads.
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        found = header_tensors(decode(stream.read(length)), 8 + length, path.stat().st_size, path.name)
    if set(found) != {t.name for t in tensors}:
        raise ValueError(f"Output inventory mismatch in {path}")


def write_json(path: Path, value: JSON) -> None:
    """Write human-readable output metadata."""
    path.write_text(json.dumps(value, indent=2) + "\n")


def assets(remote: Source) -> dict[str, bytes]:
    """Fetch tokenizer assets; Kimi code stays beside the slice, never imported by pLLM."""
    result: dict[str, bytes] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "generation_config.json",
    ):
        try:
            result[name] = remote.metadata(name)
        except FileNotFoundError:
            pass
    if (
        "tokenizer_config.json" in result
        and decode(result["tokenizer_config.json"]).get("tokenizer_class") == "TikTokenTokenizer"
    ):
        for name in ("tokenization_kimi.py", "tiktoken.model", "LICENSE"):
            result[name] = remote.metadata(name)
        if b"from .tool_declaration_ts import" in result["tokenization_kimi.py"]:
            result["tool_declaration_ts.py"] = remote.metadata("tool_declaration_ts.py")
    if "tokenizer_config.json" not in result or not (
        {"tokenizer.json", "vocab.json", "tiktoken.model"} & result.keys()
    ):
        raise ValueError("Checkpoint lacks supported tokenizer assets")
    return result


def extract(
    remote: Source,
    output: Path,
    config: dict[str, JSON],
    tensors: list[Tensor],
    manifest: dict[str, JSON],
    limit: int,
    tokenizer: dict[str, bytes],
) -> None:
    """Publish only a complete slice; leave a marked partial directory on failure."""
    staging = output.with_name(output.name + ".partial")
    if output.exists() or staging.exists():
        raise ValueError(f"Choose an unused --output (or inspect/remove the partial directory): {output}")
    staging.mkdir(parents=True)
    groups = shard_groups(tensors, limit)
    index: dict[str, JSON] = {}
    # Independent output files bound concurrency to four 8 MiB read buffers.
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures: list[Future[None]] = []
        for i, group in enumerate(groups):
            name = f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors"
            print(f"Writing {name}: {sum(t.size for t in group)} payload bytes", flush=True)
            futures.append(workers.submit(write_shard, remote, staging / name, group))
            index.update({t.name: name for t in group})
        for future in futures:
            future.result()
    for name, data in tokenizer.items():
        (staging / name).write_bytes(data)
    write_json(staging / "config.json", config)
    write_json(
        staging / "model.safetensors.index.json",
        {"metadata": {"total_size": sum(t.size for t in tensors)}, "weight_map": index},
    )
    manifest["fetched_payload_bytes"] = remote.payload_bytes
    manifest["fetched_metadata_bytes"] = remote.metadata_bytes
    manifest["tensors"] = [mapping(asdict(t)) for t in tensors]
    write_json(staging / "slice-manifest.json", manifest)
    staging.rename(output)


def main() -> int:
    """Plan or create one model slice from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="Hub owner/name; requires --revision")
    parser.add_argument("--revision", help="Immutable 40-character Hub commit SHA")
    parser.add_argument("--source", type=Path, help="Local checkpoint directory instead of --repo/--revision")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", help="Ordered source IDs, comma-separated; default first/middle/last two")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--include-mtp", action="store_true")
    parser.add_argument("--dequantize", choices=["bf16"])
    parser.add_argument(
        "--max-shard-bytes", type=int, default=2 * 1024**3, help="Payload target; a larger single tensor stays intact"
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.max_shard_bytes <= 0:
        parser.error("Use a positive --max-shard-bytes")
    remote: Source
    if args.source is not None:
        if args.repo is not None or args.revision is not None:
            parser.error("Use either --source or --repo/--revision")
        remote = Local(args.source)
        origin: dict[str, JSON] = {"source": str(args.source.resolve())}
    else:
        if re.fullmatch(r"[0-9a-f]{40}", args.revision or "") is None:
            parser.error("--revision must be an immutable lowercase 40-character commit SHA, not a branch/tag")
        if re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo or "") is None:
            parser.error("Use --repo owner/name, or --source DIR")
        remote = Remote(args.repo, args.revision)
        origin = {"repo": args.repo, "revision": args.revision}
    config, layers = rewrite_config(decode(remote.metadata("config.json")), args.layers, args.text_only)
    weights = mapping(decode(remote.metadata("model.safetensors.index.json"))["weight_map"])
    tensors = plan(remote, weights, selected_names(weights, layers))
    problems = dependency_problems(config, tensors, args.include_mtp, args.dequantize)
    tokenizer = assets(remote)
    manifest: dict[str, JSON] = {
        **origin,
        "layer_map": {str(src): dst for dst, src in enumerate(layers)},
        "config": config,
        "payload_bytes": sum(t.size for t in tensors),
        "metadata_bytes": remote.metadata_bytes,
        "tokenizer_bytes": sum(map(len, tokenizer.values())),
        "dependency_problems": list(problems),
        "decisions": [
            "text decoder only",
            "MTP excluded",
            "all selected-layer tensors retained",
            "quantized tensors kept as stored",
        ],
        "gap_overfetch_bytes": 0,
        "conversion": None,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if problems:
        return 2
    if not args.plan_only:
        extract(remote, args.output, config, tensors, manifest, args.max_shard_bytes, tokenizer)
        print(f"Complete: {args.output}; fetched payload={remote.payload_bytes}, metadata={remote.metadata_bytes}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"model_slice: {error}")
