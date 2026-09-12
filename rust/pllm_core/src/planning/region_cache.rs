//! Bounded memoization of locally derived, owned compiler data.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::thread::LocalKey;

/// Keys must include every input to the pass. Only successful local derivations
/// enter the cache: deserialized witnesses never do. Returned values are owned
/// clones, so callers cannot mutate a later validation's expected result.
pub(super) fn memoized<T: Clone + 'static>(
    cache: &'static LocalKey<RefCell<VecDeque<(String, T)>>>,
    key: String,
    derive: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    if let Some(result) = cache.with(|cache| {
        let mut cache = cache.borrow_mut();
        let index = cache.iter().position(|(stored, _)| stored == &key)?;
        let entry = cache.remove(index)?;
        let result = entry.1.clone();
        cache.push_back(entry);
        Some(result)
    }) { return Ok(result); }
    let result = derive()?;
    cache.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() == 8 { cache.pop_front(); }
        cache.push_back((key, result.clone()));
    });
    Ok(result)
}
