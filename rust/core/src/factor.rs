//! `FactorModel`: base rate times the multipliers whose rules match the inputs.

use std::collections::HashMap;

use crate::error::{Result, invalid};
use crate::value::{Inputs, Value};

#[derive(Debug, Clone)]
struct ExactMatchRule {
    exact_value: Vec<u8>,
    multiplier: f64,
}

#[derive(Debug, Clone)]
struct NumericBandRule {
    min_val: f64,
    max_val: f64,
    multiplier: f64,
}

#[derive(Debug, Clone, Default)]
pub struct FactorModel {
    base_rate: f64,
    // Rules keyed by field name; within a field they apply in the order they were added.
    exact_rules: HashMap<Vec<u8>, Vec<ExactMatchRule>>,
    band_rules: HashMap<Vec<u8>, Vec<NumericBandRule>>,
}

impl FactorModel {
    pub fn new(initial_base_rate: f64) -> Result<Self> {
        if !initial_base_rate.is_finite() {
            return invalid("base_rate must be finite");
        }
        Ok(FactorModel { base_rate: initial_base_rate, ..Default::default() })
    }

    pub fn set_base_rate(&mut self, rate: f64) -> Result<()> {
        if !rate.is_finite() {
            return invalid("base_rate must be finite");
        }
        self.base_rate = rate;
        Ok(())
    }

    pub fn add_multiplier(&mut self, field_name: &[u8], exact_value: &[u8], multiplier: f64) -> Result<()> {
        if field_name.is_empty() {
            return invalid("field_name cannot be empty");
        }
        if !multiplier.is_finite() {
            return invalid("multiplier must be finite");
        }
        self.exact_rules
            .entry(field_name.to_vec())
            .or_default()
            .push(ExactMatchRule { exact_value: exact_value.to_vec(), multiplier });
        Ok(())
    }

    pub fn add_numeric_band_multiplier(&mut self, field_name: &[u8], min_val: f64, max_val: f64, multiplier: f64) -> Result<()> {
        if field_name.is_empty() {
            return invalid("field_name cannot be empty");
        }
        if !min_val.is_finite() || !max_val.is_finite() || !multiplier.is_finite() {
            return invalid("band bounds and multiplier must be finite");
        }
        if min_val > max_val {
            return invalid("band min_val cannot exceed max_val");
        }
        self.band_rules
            .entry(field_name.to_vec())
            .or_default()
            .push(NumericBandRule { min_val, max_val, multiplier });
        Ok(())
    }

    /// Walks the inputs in key order (the C++ `std::map` order) and multiplies in every
    /// matching rule, so the floating-point product is formed in the same order.
    pub fn calculate(&self, inputs: &Inputs) -> f64 {
        let mut final_rate = self.base_rate;
        for (field_name, value) in inputs {
            match value {
                Value::Str(s) => {
                    if let Some(rules) = self.exact_rules.get(field_name) {
                        for rule in rules {
                            if *s == rule.exact_value {
                                final_rate *= rule.multiplier;
                            }
                        }
                    }
                }
                Value::Num(v) => {
                    if let Some(rules) = self.band_rules.get(field_name) {
                        for rule in rules {
                            if *v >= rule.min_val && *v <= rule.max_val {
                                final_rate *= rule.multiplier;
                            }
                        }
                    }
                }
            }
        }
        final_rate
    }
}
