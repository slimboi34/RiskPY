//! The C++ core's `DynamicField = std::variant<double, std::string>` and the input
//! map `std::map<std::string, DynamicField>`.
//!
//! Strings are kept as bytes, as `std::string` kept them: the Python layer may pass
//! `bytes`, which need not be UTF-8. A `BTreeMap` iterates in byte order, which is
//! `std::map<std::string, …>`'s order (`char_traits<char>` compares as unsigned char),
//! and `FactorModel::calculate` multiplies in that order, so the rounding matches.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    Str(Vec<u8>),
}

pub type Inputs = BTreeMap<Vec<u8>, Value>;
