//! Argument conversion with the C++ core's (pybind11 3's) rules, so every call that
//! worked against 0.8.0 still works, and every call it refused is still refused with a
//! `TypeError`:
//!
//! * `float` arguments take anything with `__float__` or `__index__` (int, bool, NumPy
//!   scalars, `Decimal`, `Fraction`), not `str`;
//! * `int` arguments refuse floats, take anything `int()` accepts through `__index__`
//!   or `__int__` (so `Decimal("5.7")` is 5), and refuse values outside the C++ type's
//!   range (`int`: ±2^31; the `seed`'s `unsigned int`: 0 … 2^32 − 1);
//! * `str` arguments take `str` (strict UTF-8), `bytes` and `bytearray`;
//! * list arguments take any sequence but `str`/`bytes`, plus generators, sets, dict
//!   views, `map` and `zip` objects; not `dict`, not plain iterators;
//! * the `inputs` mappings take a `dict` or any mapping with `.items()`, whose values
//!   are numbers or strings: a `float` or `str` keeps its type, anything else is tried
//!   as a number.
//!
//! PyO3's own conversions differ in the corners (an out-of-range int is an
//! `OverflowError`, a NumPy array is not a sequence), hence these wrappers.

use std::collections::BTreeMap;
use std::os::raw::c_long;

use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyFloat, PyFrozenSet, PySet, PyString, PyType};

use riskpy_core::{Inputs, Value};

fn incompatible(expected: &str, obj: &Bound<'_, PyAny>) -> PyErr {
    let got = obj.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "?".into());
    PyTypeError::new_err(format!("incompatible function arguments: expected {expected}, got {got}"))
}

/// Clears a pending Python error, if any.
fn clear_error() {
    // SAFETY: called with the GIL held (every caller holds a Bound).
    unsafe { ffi::PyErr_Clear() };
}

fn error_pending() -> bool {
    // SAFETY: as above.
    unsafe { !ffi::PyErr_Occurred().is_null() }
}

/// pybind11's `type_caster<double>::load(src, convert = true)`.
pub fn load_f64(obj: &Bound<'_, PyAny>) -> Option<f64> {
    // SAFETY: obj is a live object and the GIL is held.
    unsafe {
        let v = ffi::PyFloat_AsDouble(obj.as_ptr());
        if v == -1.0 && error_pending() {
            clear_error();
            if ffi::PyNumber_Check(obj.as_ptr()) != 0 {
                let tmp = ffi::PyNumber_Float(obj.as_ptr());
                clear_error();
                if tmp.is_null() {
                    return None;
                }
                let v = ffi::PyFloat_AsDouble(tmp);
                ffi::Py_DECREF(tmp);
                if v == -1.0 && error_pending() {
                    clear_error();
                    return None;
                }
                return Some(v);
            }
            return None;
        }
        Some(v)
    }
}

/// The value of an exact int object as a C long, or None (error cleared).
fn long_of(ptr: *mut ffi::PyObject) -> Option<c_long> {
    // SAFETY: ptr is a live object and the GIL is held.
    unsafe {
        let v = ffi::PyLong_AsLong(ptr);
        if v == -1 && error_pending() {
            clear_error();
            return None;
        }
        Some(v)
    }
}

/// pybind11's integer caster for a signed C++ `int` (convert = true): floats refused,
/// then `PyLong_AsLong` (which uses `__index__`), falling back to `int(obj)` when that
/// fails on a number-like object, and a range check.
// `c_long` is 64-bit on Linux and macOS but 32-bit on Windows, so the i64::from calls
// are only "useless" on some targets.
#[allow(clippy::useless_conversion)]
pub fn load_i32(obj: &Bound<'_, PyAny>) -> Option<i32> {
    if obj.is_instance_of::<PyFloat>() {
        return None;
    }
    let ptr = obj.as_ptr();
    let wide: i64 = match long_of(ptr) {
        Some(v) => i64::from(v),
        None => {
            // SAFETY: ptr is live, GIL held.
            unsafe {
                if ffi::PyNumber_Check(ptr) == 0 {
                    return None;
                }
                let tmp = ffi::PyNumber_Long(ptr);
                clear_error();
                if tmp.is_null() {
                    return None;
                }
                let v = long_of(tmp);
                ffi::Py_DECREF(tmp);
                i64::from(v?)
            }
        }
    };
    i32::try_from(wide).ok()
}

/// pybind11's integer caster for a C++ `unsigned int` (the simulator's seed):
/// `PyLong_AsUnsignedLong` takes only real ints, so everything else goes through
/// `int(obj)`; negative or too large is refused.
pub fn load_u32(obj: &Bound<'_, PyAny>) -> Option<u32> {
    if obj.is_instance_of::<PyFloat>() {
        return None;
    }
    fn unsigned_of(ptr: *mut ffi::PyObject) -> Option<u64> {
        // SAFETY: ptr live, GIL held.
        unsafe {
            let v = ffi::PyLong_AsUnsignedLongLong(ptr);
            if v == u64::MAX && error_pending() {
                clear_error();
                return None;
            }
            Some(v)
        }
    }
    let ptr = obj.as_ptr();
    // SAFETY: ptr live, GIL held.
    let is_int = unsafe { ffi::PyLong_Check(ptr) != 0 };
    let wide = if is_int { unsigned_of(ptr) } else { None };
    let wide = match wide {
        Some(v) => v,
        None => unsafe {
            if ffi::PyNumber_Check(ptr) == 0 {
                return None;
            }
            let tmp = ffi::PyNumber_Long(ptr);
            clear_error();
            if tmp.is_null() {
                return None;
            }
            let v = unsigned_of(tmp);
            ffi::Py_DECREF(tmp);
            v?
        },
    };
    u32::try_from(wide).ok()
}

/// pybind11's `std::string` caster: str (strict UTF-8), bytes or bytearray.
pub fn load_bytes(obj: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
    if let Ok(s) = obj.cast::<PyString>() {
        return match s.to_str() {
            Ok(text) => Some(text.as_bytes().to_vec()),
            Err(_) => None,
        };
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Some(b.as_bytes().to_vec());
    }
    if let Ok(b) = obj.cast::<PyByteArray>() {
        return Some(b.to_vec());
    }
    None
}

fn generator_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static GENERATOR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    GENERATOR.import(py, "types", "GeneratorType")
}

/// The elements pybind11 3's `list_caster` would iterate, or None if it refuses the
/// object. Generators, sets and the named iterables are exhausted first, as
/// `tuple(obj)` would.
fn sequence_items<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    let py = obj.py();
    let ptr = obj.as_ptr();
    // SAFETY: ptr live, GIL held.
    let is_sequence = unsafe { ffi::PySequence_Check(ptr) != 0 };
    let accepted = if is_sequence {
        !obj.is_instance_of::<PyString>() && !obj.is_instance_of::<PyBytes>()
    } else {
        obj.is_instance(generator_type(py)?.as_any())?
            || obj.is_instance_of::<PySet>()
            || obj.is_instance_of::<PyFrozenSet>()
            || matches!(
                obj.get_type().name()?.to_str()?,
                "dict_keys" | "dict_values" | "dict_items" | "map" | "zip"
            )
    };
    if !accepted {
        return Ok(None);
    }
    let mut items = Vec::new();
    for item in obj.try_iter()? {
        items.push(item?);
    }
    Ok(Some(items))
}

/// One value of an `inputs` mapping: pybind11's `variant<double, std::string>` caster,
/// which tries both alternatives without conversion first (so a float stays a float
/// and a str a str), then the double with conversion.
fn load_value(obj: &Bound<'_, PyAny>) -> Option<Value> {
    if obj.is_instance_of::<PyFloat>() {
        return load_f64(obj).map(Value::Num);
    }
    if let Some(b) = load_bytes(obj) {
        return Some(Value::Str(b));
    }
    load_f64(obj).map(Value::Num)
}

/// pybind11 3's `map_caster` for `std::map<std::string, DynamicField>`: a dict, or any
/// mapping with `.items()` (copied into a dict first). Keys go through the string
/// caster; the first occurrence of a key wins, as `std::map::emplace` keeps it.
fn load_inputs(obj: &Bound<'_, PyAny>) -> PyResult<Option<Inputs>> {
    let py = obj.py();
    let dict: Bound<'_, PyDict> = if let Ok(d) = obj.cast::<PyDict>() {
        d.clone()
    } else {
        // SAFETY: ptr live, GIL held.
        let is_mapping = unsafe {
            ffi::PyMapping_Check(obj.as_ptr()) != 0 && ffi::PyObject_HasAttrString(obj.as_ptr(), c"items".as_ptr()) != 0
        };
        if !is_mapping {
            return Ok(None);
        }
        let items = obj.call_method0("items")?;
        py.get_type::<PyDict>().call1((items,))?.cast_into::<PyDict>()?
    };
    let mut out = BTreeMap::new();
    for (k, v) in dict.iter() {
        let (Some(key), Some(value)) = (load_bytes(&k), load_value(&v)) else {
            return Ok(None);
        };
        out.entry(key).or_insert(value);
    }
    Ok(Some(out))
}

// ---------------------------------------------------------------------------------
// Newtypes used as #[pymethods] argument types.

/// A C++ `double` argument.
#[derive(Clone, Copy, Debug)]
pub struct CFloat(pub f64);

/// A C++ `int` argument.
#[derive(Clone, Copy, Debug)]
pub struct CInt(pub i32);

/// A C++ `unsigned int` argument.
#[derive(Clone, Copy, Debug)]
pub struct CUInt(pub u32);

/// A C++ `std::string` argument (raw bytes, as `std::string` holds them).
#[derive(Clone, Debug, Default)]
pub struct CStr(pub Vec<u8>);

/// A C++ `std::vector<double>` argument.
#[derive(Clone, Debug)]
pub struct CFloats(pub Vec<f64>);

/// A C++ `std::map<std::string, DynamicField>` argument.
#[derive(Clone, Debug)]
pub struct CInputs(pub Inputs);

/// A C++ `std::vector<std::map<std::string, DynamicField>>` argument.
#[derive(Clone, Debug)]
pub struct CInputsList(pub Vec<Inputs>);

impl<'py> FromPyObject<'_, 'py> for CFloat {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        load_f64(&obj).map(CFloat).ok_or_else(|| incompatible("a float", &obj))
    }
}

impl<'py> FromPyObject<'_, 'py> for CInt {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        load_i32(&obj).map(CInt).ok_or_else(|| incompatible("an int from -2**31 to 2**31 - 1", &obj))
    }
}

impl<'py> FromPyObject<'_, 'py> for CUInt {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        load_u32(&obj).map(CUInt).ok_or_else(|| incompatible("an int from 0 to 2**32 - 1", &obj))
    }
}

impl<'py> FromPyObject<'_, 'py> for CStr {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        load_bytes(&obj).map(CStr).ok_or_else(|| incompatible("a str", &obj))
    }
}

impl<'py> FromPyObject<'_, 'py> for CFloats {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let Some(items) = sequence_items(&obj)? else {
            return Err(incompatible("a sequence of floats", &obj));
        };
        let mut out = Vec::with_capacity(items.len());
        for item in &items {
            out.push(load_f64(item).ok_or_else(|| incompatible("a sequence of floats", &obj))?);
        }
        Ok(CFloats(out))
    }
}

impl<'py> FromPyObject<'_, 'py> for CInputs {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        load_inputs(&obj)?
            .map(CInputs)
            .ok_or_else(|| incompatible("a mapping of str to float or str", &obj))
    }
}

impl<'py> FromPyObject<'_, 'py> for CInputsList {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let expected = "a sequence of mappings of str to float or str";
        let Some(items) = sequence_items(&obj)? else {
            return Err(incompatible(expected, &obj));
        };
        let mut out = Vec::with_capacity(items.len());
        for item in &items {
            out.push(load_inputs(item)?.ok_or_else(|| incompatible(expected, &obj))?);
        }
        Ok(CInputsList(out))
    }
}

/// `std::string` → `str`: strict UTF-8, so bytes that are not UTF-8 raise
/// `UnicodeDecodeError` with Python's own message, as pybind11's cast did.
pub fn to_py_str<'py>(py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyString>> {
    PyString::from_bytes(py, bytes)
}

/// The inputs map as the dict pybind11 handed to a Python callback: keys in map
/// order, floats and strs.
pub fn inputs_to_dict<'py>(py: Python<'py>, inputs: &Inputs) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in inputs {
        let key = to_py_str(py, k)?;
        match v {
            Value::Num(x) => d.set_item(key, *x)?,
            Value::Str(s) => d.set_item(key, to_py_str(py, s)?)?,
        }
    }
    Ok(d)
}
