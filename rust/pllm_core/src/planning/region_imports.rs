//! Execution-boundary storage imports, separate from source semantics.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use super::region_program::{AccessMode, AffineIndex, Extent, Function, IndexTerm, ScalarType, SizeParameter, Type};
use super::tensor_access::{self, AccessGraph, StorageView};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct KernelAllocation { pub name: String, pub device: String, pub bytes: u64 }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum TensorImport {
    Exported { access: AccessGraph },
    Native { declaration: Function },
    Borrowed { declaration: Function, field: u64 },
    NativeKernel { declaration: Function, implementation: String, device: String, allocations: Vec<KernelAllocation>, host_reads: bool },
    FunctionalKernel { declaration: Function, implementation: String, device: String, allocations: Vec<KernelAllocation> },
    Compiled { source_access: AccessGraph, backend: String, code_digest: String,
        allocations: Vec<KernelAllocation> },
}

pub(super) fn tensor_bytes(type_: &Type) -> Result<u64, String> {
    let Type::Tensor { shape, dtype, .. } = type_ else {
        return Err("native storage imports currently require pure all-tensor functions".into());
    };
    StorageView { name: String::new(), storage: 0, shape: shape.clone(),
        stride: vec![], offset: 0, dtype: dtype.clone() }.logical_bytes()
}

/// NativePrimitive recursively copies every tensor leaf, including repeated
/// views. A bank is an existing immutable value container, not a new storage
/// owner. Use declared maxima for symbolic extents; this can overestimate
/// jointly constrained extents but never substitutes the tuning sample size.
pub(super) fn native_copy_bytes<'a>(type_: &'a Type, parameters: &[SizeParameter])
    -> Result<Option<(&'a str, u64)>, String>
{
    match type_ {
        Type::Tensor { device, .. } => Ok(Some((device, tensor_bytes(type_)?))),
        Type::Scalar { .. } => Ok(None),
        Type::Bank { element, size } => {
            let count = match size {
                Extent::Fixed(count) => *count,
                Extent::Parameter(name) => parameters.iter().find(|p| &p.name == name)
                    .ok_or("native bank copy has an undeclared extent")?.maximum,
            };
            native_copy_bytes(element, parameters)?.map(|(device, bytes)| {
                bytes.checked_mul(count).map(|bytes| (device, bytes))
                    .ok_or_else(|| "native bank copy envelope overflows u64".into())
            }).transpose()
        },
    }
}

fn validate_native_port(type_: &Type) -> Result<(), String> {
    match type_ {
        Type::Tensor { .. } => tensor_bytes(type_).map(|_| ()),
        Type::Scalar { .. } => Ok(()),
        Type::Bank { element, .. } => validate_native_port(element),
    }
}

impl TensorImport {
    pub fn name(&self) -> &str {
        match self {
            Self::Exported { access } => &access.source_digest,
            Self::Native { declaration } | Self::NativeKernel { declaration, .. }
                | Self::FunctionalKernel { declaration, .. } | Self::Borrowed { declaration, .. } => &declaration.name,
            Self::Compiled { source_access, .. } => &source_access.source_digest,
        }
    }

    pub fn validate(&self, function: &Function) -> Result<(), String> {
        if !matches!(self, Self::Native { .. } | Self::NativeKernel { .. } | Self::Borrowed { .. }) && !function.resources.is_empty() {
            return Err("storage import requires a pure tensor function".into());
        }
        match self {
            Self::Borrowed { declaration, field } => {
                if declaration != function || *field > i64::MAX as u64
                    || declaration.resources.len() != 1 || declaration.outputs.len() != 1
                    || !matches!(declaration.outputs[0], Type::Tensor { .. })
                    || declaration.inputs.iter().any(|t| !matches!(t, Type::Scalar { name: ScalarType::Index }))
                    || declaration.accesses.len() != 1
                { return Err("tensor borrowing requires one typed field and index coordinates".into()); }
                tensor_bytes(&declaration.outputs[0])?;
                let access = &declaration.accesses[0];
                if access.resource != declaration.resources[0] || access.mode != AccessMode::Read
                    || access.ranges.len() != declaration.inputs.len()
                    || access.ranges.iter().enumerate().any(|(argument, range)|
                        range.lower != (AffineIndex { constant: 0, terms: vec![IndexTerm { argument, coefficient: 1 }] })
                        || range.upper != (AffineIndex { constant: 1, terms: vec![IndexTerm { argument, coefficient: 1 }] }))
                { return Err("tensor borrowing must read exactly its original resource cell".into()); }
            },
            Self::Compiled { source_access, backend, code_digest, allocations } => {
                // The source access graph proves the tensor ABI, not backend
                // allocation sizes. The local binder replays the compiler.
                Self::Exported { access: source_access.clone() }.validate(function)?;
                if backend.is_empty() || code_digest.len() != 64
                    || !code_digest.bytes().all(|b| b.is_ascii_hexdigit())
                { return Err("compiled tensor import needs a backend/code identity".into()); }
                let values = tensor_access::validate(source_access)?;
                // Readonly inputs keep their validated source strides. Only
                // returned ports require the canonical independent-copy layout.
                for name in &source_access.outputs {
                    let value = values[name.as_str()];
                    let mut expected = 1u64;
                    for (&size, &stride) in value.shape.iter().zip(&value.stride).rev() {
                        if stride != expected { return Err("compiled copy ABI requires canonical contiguous ports".into()); }
                        expected = expected.checked_mul(size.max(1)).ok_or("compiled port size overflows")?;
                    }
                    // Source slices may have nonzero offsets. The compiled
                    // ABI guards shape/stride, accepts unaligned input data
                    // pointers, and independently copies every returned port.
                }
                let devices: BTreeSet<_> = source_access.storages.iter().map(|s| s.device.as_str()).collect();
                let mut names = BTreeSet::new();
                for buffer in allocations {
                    if buffer.name.is_empty() || !names.insert(&buffer.name) || !devices.contains(buffer.device.as_str()) {
                        return Err("compiled allocations require unique names and source devices".into());
                    }
                }
            },
            Self::Native { declaration } | Self::NativeKernel { declaration, .. }
                | Self::FunctionalKernel { declaration, .. } => {
                if declaration != function {
                    return Err("native storage import differs from its source declaration".into());
                }
                for type_ in declaration.inputs.iter().chain(&declaration.outputs) {
                    validate_native_port(type_)?;
                }
                if let Self::NativeKernel { implementation, device, allocations, .. }
                    | Self::FunctionalKernel { implementation, device, allocations, .. } = self {
                    if implementation.is_empty() || device.is_empty() {
                        return Err("native kernel requires a local implementation and execution device".into());
                    }
                    let mut names = BTreeSet::new();
                    for buffer in allocations {
                        if buffer.name.is_empty() || !names.insert(&buffer.name) || &buffer.device != device {
                            return Err("native kernel allocations require unique names on the execution device".into());
                        }
                    }
                    for type_ in declaration.inputs.iter().chain(&declaration.outputs) {
                        if let Some(Type::Tensor { device: port_device, .. }) = super::region_storage::tensor_leaf(type_) {
                            if port_device != device { return Err("native kernel ports must use its execution device".into()); }
                        }
                    }
                    if matches!(self, Self::FunctionalKernel { .. }) {
                        for type_ in declaration.inputs.iter().chain(&declaration.outputs) { tensor_bytes(type_)?; }
                        let results = declaration.outputs.iter().try_fold(0_u64, |total, type_| {
                            total.checked_add(tensor_bytes(type_)?).ok_or("functional result bytes overflow".to_owned())
                        })?;
                        let allocated = allocations.iter().try_fold(0_u64, |total, item|
                            total.checked_add(item.bytes).ok_or("functional allocation bytes overflow"))?;
                        if allocated < results { return Err("functional kernel allocations must include every owned result".into()); }
                    }
                }
            },
            Self::Exported { access } => {
                let values = tensor_access::validate(access)?;
                if function.inputs.len() != access.inputs.len() || function.outputs.len() != access.outputs.len() {
                    return Err("tensor storage import has a different function ABI".into());
                }
                for (type_, name) in function.inputs.iter().zip(&access.inputs)
                    .chain(function.outputs.iter().zip(&access.outputs))
                {
                    let view = values[name.as_str()];
                    if let Type::Tensor { shape, dtype, device } = type_ {
                        if shape == &view.shape && dtype == &view.dtype && device == &access.storages[view.storage].device {
                            continue;
                        }
                    }
                    return Err("tensor storage import differs from its source function signature".into());
                }
            },
        }
        Ok(())
    }
}
