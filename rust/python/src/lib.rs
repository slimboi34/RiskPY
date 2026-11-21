//! `riskpy.cpp_underwriter`: the compiled core of RiskPY, as a PyO3 extension.
//!
//! The module keeps the C++ core's name, classes, methods, argument names, defaults and
//! exceptions, so the Python layer and user code import and call it unchanged. The
//! work is done by the `riskpy-core` crate; this file only converts arguments (with the
//! old pybind11 rules, see `conv`) and maps errors:
//! `ValueError` for a refused argument, `RuntimeError` for a failed export,
//! `MemoryError` for an allocation that cannot be made.
//!
//! Every `simulate_*` method and the FFT functions release the GIL while they compute.

mod conv;

use std::io::Write as _;

use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::PyTraverseError;

use riskpy_core as core;
use riskpy_core::xlsx::ColumnMapping;

use conv::{CFloat, CFloats, CInputs, CInputsList, CInt, CStr, CUInt, inputs_to_dict, load_f64, to_py_str};


fn err(e: core::Error) -> PyErr {
    match e {
        core::Error::Value(m) => PyValueError::new_err(m),
        core::Error::Runtime(m) => PyRuntimeError::new_err(m),
        core::Error::Memory(m) => PyMemoryError::new_err(m),
    }
}

trait OrPy<T> {
    fn py_err(self) -> PyResult<T>;
}

impl<T> OrPy<T> for core::Result<T> {
    fn py_err(self) -> PyResult<T> {
        self.map_err(err)
    }
}

/// Writes a line to the process's standard output (file descriptor 1), as the C++
/// core's `std::cout << … << std::endl` did; a closed stdout is ignored.
fn print_stdout(line: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// The same on standard error, for `std::cerr`.
fn print_stderr(line: &str) {
    let mut out = std::io::stderr().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

// =====================================================================================
// Field, ExcelExporter, RiskEngine

/// A named, labelled input field of a rating engine.
#[pyclass(module = "riskpy.cpp_underwriter", name = "Field", subclass, weakref, skip_from_py_object)]
#[derive(Clone)]
pub struct Field {
    name: Vec<u8>,
    label: Vec<u8>,
    kind: Vec<u8>,
}

#[pymethods]
impl Field {
    #[new]
    #[pyo3(signature = (name = CStr::default(), label = CStr::default(), r#type = CStr::default()),
           text_signature = "(name='', label='', type='')")]
    fn new(name: CStr, label: CStr, r#type: CStr) -> Self {
        Field { name: name.0, label: label.0, kind: r#type.0 }
    }

    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        to_py_str(py, &self.name)
    }

    #[setter]
    fn set_name(&mut self, value: CStr) {
        self.name = value.0;
    }

    #[getter]
    fn label<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        to_py_str(py, &self.label)
    }

    #[setter]
    fn set_label(&mut self, value: CStr) {
        self.label = value.0;
    }

    #[getter(r#type)]
    fn kind<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        to_py_str(py, &self.kind)
    }

    #[setter(r#type)]
    fn set_kind(&mut self, value: CStr) {
        self.kind = value.0;
    }
}

/// Maps rating fields to spreadsheet columns and writes quotes to an .xlsx file.
///
/// Every export starts from a blank workbook with one sheet, "Sheet1": the `template`
/// argument is kept for compatibility and, as in every earlier release, not read.
#[pyclass(module = "riskpy.cpp_underwriter", name = "ExcelExporter", subclass, weakref)]
pub struct ExcelExporter {
    #[allow(dead_code)] // kept, and ignored, as the C++ core kept it
    template_name: Vec<u8>,
    mapping: ColumnMapping,
}

#[pymethods]
impl ExcelExporter {
    #[new]
    #[pyo3(signature = (template))]
    fn new(template: CStr) -> Self {
        ExcelExporter { template_name: template.0, mapping: ColumnMapping::new() }
    }

    /// map_column(col, field_name, label): put `field_name` (or the premium, for
    /// "premium_output") in column `col` under the header `label`.
    #[pyo3(signature = (col, field_name, label, /))]
    fn map_column(&mut self, col: CStr, field_name: CStr, label: CStr) {
        self.mapping.insert(col.0, (field_name.0, label.0));
    }
}

/// A rating engine: fields, a Python (or FactorModel) pricing callback, and an
/// attached ExcelExporter.
#[pyclass(module = "riskpy.cpp_underwriter", name = "RiskEngine", subclass, weakref)]
pub struct RiskEngine {
    fields: Vec<Field>,
    logic: Option<Py<PyAny>>,
    exporter: Option<Py<ExcelExporter>>,
    last_inputs: core::Inputs,
    last_premium: f64,
}

#[pymethods]
impl RiskEngine {
    #[new]
    fn new() -> Self {
        RiskEngine { fields: Vec::new(), logic: None, exporter: None, last_inputs: core::Inputs::new(), last_premium: 0.0 }
    }

    #[pyo3(signature = (field, /))]
    fn add_field(&mut self, field: PyRef<'_, Field>) {
        self.fields.push((*field).clone());
    }

    /// A list of copies of the fields added so far.
    fn get_fields(&self) -> Vec<Field> {
        self.fields.clone()
    }

    /// set_math_logic(logic): a callable taking the inputs dict and returning the
    /// premium; None removes it (execute then returns 0.0).
    #[pyo3(signature = (logic, /))]
    fn set_math_logic(&mut self, logic: &Bound<'_, PyAny>) -> PyResult<()> {
        if logic.is_none() {
            self.logic = None;
        } else if logic.is_callable() {
            self.logic = Some(logic.clone().unbind());
        } else {
            let got = logic.get_type().name()?.to_string();
            return Err(PyTypeError::new_err(format!(
                "incompatible function arguments: expected a callable or None, got {got}"
            )));
        }
        Ok(())
    }

    /// attach_exporter(exporter): the engine keeps the exporter alive; None detaches it.
    #[pyo3(signature = (exporter, /))]
    fn attach_exporter(&mut self, exporter: Option<Py<ExcelExporter>>) {
        self.exporter = exporter;
    }

    /// execute(inputs): the premium from the math logic (0.0 without one). The inputs
    /// and the premium are kept for export_to_excel.
    #[pyo3(signature = (inputs, /))]
    fn execute(slf: &Bound<'_, Self>, inputs: CInputs) -> PyResult<f64> {
        let py = slf.py();
        // Borrow only around the bookkeeping, so the callback may use the engine.
        let logic = {
            let mut me = slf.borrow_mut();
            me.last_inputs = inputs.0.clone();
            me.logic.as_ref().map(|f| f.clone_ref(py))
        };
        let Some(logic) = logic else {
            return Ok(0.0);
        };
        let ret = logic.call1(py, (inputs_to_dict(py, &inputs.0)?,))?;
        let ret = ret.bind(py);
        let premium = load_f64(ret).ok_or_else(|| {
            let ty = ret.get_type().repr().map(|r| r.to_string()).unwrap_or_else(|_| "?".into());
            PyRuntimeError::new_err(format!("Unable to cast Python instance of type {ty} to float: the math logic must return a number"))
        })?;
        slf.borrow_mut().last_premium = premium;
        Ok(premium)
    }

    /// export_to_excel(filename): the last executed quote, labels in row 1 and values in row 2.
    #[pyo3(signature = (filename, /))]
    fn export_to_excel(&self, py: Python<'_>, filename: CStr) -> PyResult<()> {
        let Some(exporter) = &self.exporter else {
            print_stderr("No exporter attached!");
            return Ok(());
        };
        let sheet = {
            let ex = exporter.bind(py).borrow();
            core::xlsx::single_sheet(&ex.mapping, &self.last_inputs, self.last_premium).py_err()?
        };
        core::xlsx::save(&filename.0, &sheet).py_err()?;
        print_stdout(&format!("Successfully exported data to {}", String::from_utf8_lossy(&filename.0)));
        Ok(())
    }

    /// export_batch_to_excel(filename, batch_inputs, batch_premiums): one row per quote
    /// from row 2; a quote without a premium gets 0.
    #[pyo3(signature = (filename, batch_inputs, batch_premiums, /))]
    fn export_batch_to_excel(&self, py: Python<'_>, filename: CStr, batch_inputs: CInputsList, batch_premiums: CFloats) -> PyResult<()> {
        let Some(exporter) = &self.exporter else {
            print_stderr("No exporter attached!");
            return Ok(());
        };
        let sheet = {
            let ex = exporter.bind(py).borrow();
            core::xlsx::batch_sheet(&ex.mapping, &batch_inputs.0, &batch_premiums.0).py_err()?
        };
        core::xlsx::save(&filename.0, &sheet).py_err()?;
        print_stdout(&format!("Successfully exported batch data to {}", String::from_utf8_lossy(&filename.0)));
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.logic)?;
        visit.call(&self.exporter)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        self.logic = None;
        self.exporter = None;
    }
}

// =====================================================================================
// ActuarialMath, FactorModel

/// Annuity values, loss ratio and a simplified mortality table (static methods).
#[pyclass(module = "riskpy.cpp_underwriter", name = "ActuarialMath", subclass, weakref, frozen)]
pub struct ActuarialMath;

#[pymethods]
impl ActuarialMath {
    /// Present value of an annuity-immediate: payment * (1 - (1 + rate)**-periods) / rate.
    #[staticmethod]
    #[pyo3(signature = (rate, periods, payment))]
    fn present_value(rate: CFloat, periods: CInt, payment: CFloat) -> PyResult<f64> {
        core::actuarial::present_value(rate.0, periods.0, payment.0).py_err()
    }

    /// Future value of an annuity-immediate: payment * ((1 + rate)**periods - 1) / rate.
    #[staticmethod]
    #[pyo3(signature = (rate, periods, payment))]
    fn future_value(rate: CFloat, periods: CInt, payment: CFloat) -> PyResult<f64> {
        core::actuarial::future_value(rate.0, periods.0, payment.0).py_err()
    }

    /// Incurred losses over earned premium.
    #[staticmethod]
    #[pyo3(signature = (incurred_losses, earned_premium))]
    fn calculate_loss_ratio(incurred_losses: CFloat, earned_premium: CFloat) -> PyResult<f64> {
        core::actuarial::calculate_loss_ratio(incurred_losses.0, earned_premium.0).py_err()
    }

    /// Annual mortality rate from a simplified mock of the CSO 2001 table.
    #[staticmethod]
    #[pyo3(signature = (age))]
    fn lookup_mortality_rate(age: CInt) -> f64 {
        core::actuarial::lookup_mortality_rate(age.0)
    }
}

/// Factor rating: a base rate times every matching exact-value and numeric-band multiplier.
#[pyclass(module = "riskpy.cpp_underwriter", name = "FactorModel", subclass, weakref)]
pub struct FactorModel {
    inner: core::factor::FactorModel,
}

#[pymethods]
impl FactorModel {
    #[new]
    #[pyo3(signature = (initial_base_rate = CFloat(0.0)), text_signature = "(initial_base_rate=0.0)")]
    fn new(initial_base_rate: CFloat) -> PyResult<Self> {
        Ok(FactorModel { inner: core::factor::FactorModel::new(initial_base_rate.0).py_err()? })
    }

    #[pyo3(signature = (rate))]
    fn set_base_rate(&mut self, rate: CFloat) -> PyResult<()> {
        self.inner.set_base_rate(rate.0).py_err()
    }

    /// Multiply by `multiplier` when the string input `field_name` equals `exact_value`.
    #[pyo3(signature = (field_name, exact_value, multiplier))]
    fn add_multiplier(&mut self, field_name: CStr, exact_value: CStr, multiplier: CFloat) -> PyResult<()> {
        self.inner.add_multiplier(&field_name.0, &exact_value.0, multiplier.0).py_err()
    }

    /// Multiply by `multiplier` when the numeric input `field_name` is in [min_val, max_val].
    #[pyo3(signature = (field_name, min_val, max_val, multiplier))]
    fn add_numeric_band_multiplier(&mut self, field_name: CStr, min_val: CFloat, max_val: CFloat, multiplier: CFloat) -> PyResult<()> {
        self.inner.add_numeric_band_multiplier(&field_name.0, min_val.0, max_val.0, multiplier.0).py_err()
    }

    /// The rate for a dict of inputs (str values match exact rules, numbers match bands).
    #[pyo3(signature = (inputs))]
    fn calculate(&self, inputs: CInputs) -> f64 {
        self.inner.calculate(&inputs.0)
    }
}

// =====================================================================================
// MonteCarloSimulator

/// Five actuarial simulations. A non-zero seed is reproducible on every platform; seed 0
/// draws a fresh one.
#[pyclass(module = "riskpy.cpp_underwriter", name = "MonteCarloSimulator", subclass, weakref, frozen)]
pub struct MonteCarloSimulator {
    inner: core::montecarlo::MonteCarloSimulator,
}

#[pymethods]
impl MonteCarloSimulator {
    #[new]
    #[pyo3(signature = (trials = CInt(10000), seed = CUInt(0)), text_signature = "(trials=10000, seed=0)")]
    fn new(trials: CInt, seed: CUInt) -> PyResult<Self> {
        Ok(MonteCarloSimulator { inner: core::montecarlo::MonteCarloSimulator::new(trials.0, seed.0).py_err()? })
    }

    /// P&C aggregate loss per trial: Poisson(expected_frequency) claims, each
    /// lognormal(expected_severity_mu, severity_sigma).
    #[pyo3(signature = (expected_frequency, expected_severity_mu, severity_sigma))]
    fn simulate_aggregate_loss(&self, py: Python<'_>, expected_frequency: CFloat, expected_severity_mu: CFloat, severity_sigma: CFloat) -> PyResult<Vec<f64>> {
        let sim = &self.inner;
        py.detach(|| sim.simulate_aggregate_loss(expected_frequency.0, expected_severity_mu.0, severity_sigma.0)).py_err()
    }

    /// Life: death_benefit times Binomial(policy_count, q) deaths, where
    /// q = base_mortality_rate * max(0, Normal(1, shock_volatility)), capped at 1.
    #[pyo3(signature = (policy_count, base_mortality_rate, shock_volatility, death_benefit))]
    fn simulate_life_portfolio(&self, py: Python<'_>, policy_count: CInt, base_mortality_rate: CFloat, shock_volatility: CFloat, death_benefit: CFloat) -> PyResult<Vec<f64>> {
        let sim = &self.inner;
        py.detach(|| sim.simulate_life_portfolio(policy_count.0, base_mortality_rate.0, shock_volatility.0, death_benefit.0)).py_err()
    }

    /// Terminal value of a geometric Brownian motion after time_steps unit steps.
    #[pyo3(signature = (initial_price, drift, volatility, time_steps))]
    fn simulate_economic_path(&self, py: Python<'_>, initial_price: CFloat, drift: CFloat, volatility: CFloat, time_steps: CInt) -> PyResult<Vec<f64>> {
        let sim = &self.inner;
        py.detach(|| sim.simulate_economic_path(initial_price.0, drift.0, volatility.0, time_steps.0)).py_err()
    }

    /// Health: negative-binomial claim counts (mean expected_frequency, variance
    /// m + dispersion * m**2, as a gamma-Poisson mixture) with Gamma(shape, scale) claims.
    #[pyo3(signature = (expected_frequency, dispersion, severity_shape, severity_scale))]
    fn simulate_health_claims(&self, py: Python<'_>, expected_frequency: CFloat, dispersion: CFloat, severity_shape: CFloat, severity_scale: CFloat) -> PyResult<Vec<f64>> {
        let sim = &self.inner;
        py.detach(|| sim.simulate_health_claims(expected_frequency.0, dispersion.0, severity_shape.0, severity_scale.0)).py_err()
    }

    /// Catastrophe: with probability probability_of_event, a Pareto(pareto_xm, pareto_alpha) loss.
    #[pyo3(signature = (probability_of_event, pareto_xm, pareto_alpha))]
    fn simulate_catastrophe_loss(&self, py: Python<'_>, probability_of_event: CFloat, pareto_xm: CFloat, pareto_alpha: CFloat) -> PyResult<Vec<f64>> {
        let sim = &self.inner;
        py.detach(|| sim.simulate_catastrophe_loss(probability_of_event.0, pareto_xm.0, pareto_alpha.0)).py_err()
    }
}

// =====================================================================================
// LossTriangle, ExperienceRating, ExposureRating, RateAnalyzer

/// A cumulative loss triangle (rows may be ragged) with chain-ladder projections.
#[pyclass(module = "riskpy.cpp_underwriter", name = "LossTriangle", subclass, weakref)]
pub struct LossTriangle {
    inner: core::triangle::LossTriangle,
}

#[pymethods]
impl LossTriangle {
    #[new]
    fn new() -> Self {
        LossTriangle { inner: core::triangle::LossTriangle::new() }
    }

    #[pyo3(signature = (year, cumulative_payments))]
    fn add_origin_year(&mut self, year: CInt, cumulative_payments: CFloats) -> PyResult<()> {
        self.inner.add_origin_year(year.0, cumulative_payments.0).py_err()
    }

    /// Volume-weighted age-to-age factors.
    fn get_development_factors(&self) -> PyResult<Vec<f64>> {
        self.inner.development_factors().py_err()
    }

    /// Chain-ladder ultimate losses per origin year.
    fn get_ultimate_losses(&self) -> PyResult<Vec<f64>> {
        self.inner.ultimate_losses().py_err()
    }

    /// Ultimate minus latest diagonal, floored at zero.
    fn get_ibnr_reserves(&self) -> PyResult<Vec<f64>> {
        self.inner.ibnr_reserves().py_err()
    }

    fn get_origin_year_count(&self) -> usize {
        self.inner.origin_year_count()
    }

    fn get_development_period_count(&self) -> usize {
        self.inner.development_period_count()
    }
}

/// Credibility and experience modification.
#[pyclass(module = "riskpy.cpp_underwriter", name = "ExperienceRating", subclass, weakref, frozen)]
pub struct ExperienceRating {
    inner: core::experience::ExperienceRating,
}

#[pymethods]
impl ExperienceRating {
    #[new]
    #[pyo3(signature = (k_parameter = CFloat(1082.0), ballast = CFloat(0.0)), text_signature = "(k_parameter=1082.0, ballast=0.0)")]
    fn new(k_parameter: CFloat, ballast: CFloat) -> PyResult<Self> {
        Ok(ExperienceRating { inner: core::experience::ExperienceRating::new(k_parameter.0, ballast.0).py_err()? })
    }

    #[pyo3(signature = (observed_rate, expected_rate, credibility_z))]
    fn credibility_weighted_rate(&self, observed_rate: CFloat, expected_rate: CFloat, credibility_z: CFloat) -> PyResult<f64> {
        self.inner.credibility_weighted_rate(observed_rate.0, expected_rate.0, credibility_z.0).py_err()
    }

    #[pyo3(signature = (claim_count))]
    fn calculate_credibility(&self, claim_count: CInt) -> PyResult<f64> {
        self.inner.calculate_credibility(claim_count.0).py_err()
    }

    #[pyo3(signature = (actual_losses, expected_losses))]
    fn experience_mod_factor(&self, actual_losses: CFloat, expected_losses: CFloat) -> PyResult<f64> {
        self.inner.experience_mod_factor(actual_losses.0, expected_losses.0).py_err()
    }

    #[pyo3(signature = (exposure, variance_ratio))]
    fn buhlmann_straub_credibility(&self, exposure: CFloat, variance_ratio: CFloat) -> PyResult<f64> {
        self.inner.buhlmann_straub_credibility(exposure.0, variance_ratio.0).py_err()
    }

    #[pyo3(signature = (exposure, frequency, severity))]
    fn expected_losses(&self, exposure: CFloat, frequency: CFloat, severity: CFloat) -> PyResult<f64> {
        self.inner.expected_losses(exposure.0, frequency.0, severity.0).py_err()
    }
}

/// Increased limits factors, layer pricing, burning cost and the lognormal LER.
#[pyclass(module = "riskpy.cpp_underwriter", name = "ExposureRating", subclass, weakref, frozen)]
pub struct ExposureRating {
    inner: core::exposure::ExposureRating,
}

#[pymethods]
impl ExposureRating {
    #[new]
    #[pyo3(signature = (base_limit = CFloat(100000.0), b_parameter = CFloat(1.0)), text_signature = "(base_limit=100000.0, b_parameter=1.0)")]
    fn new(base_limit: CFloat, b_parameter: CFloat) -> PyResult<Self> {
        Ok(ExposureRating { inner: core::exposure::ExposureRating::new(base_limit.0, b_parameter.0).py_err()? })
    }

    #[pyo3(signature = (target_limit))]
    fn increased_limits_factor(&self, target_limit: CFloat) -> PyResult<f64> {
        self.inner.increased_limits_factor(target_limit.0).py_err()
    }

    #[pyo3(signature = (ground_up_premium, attachment, limit))]
    fn layer_premium(&self, ground_up_premium: CFloat, attachment: CFloat, limit: CFloat) -> PyResult<f64> {
        self.inner.layer_premium(ground_up_premium.0, attachment.0, limit.0).py_err()
    }

    #[pyo3(signature = (historical_losses, attachment, limit, exposure_years))]
    fn burning_cost(&self, historical_losses: CFloats, attachment: CFloat, limit: CFloat, exposure_years: CInt) -> PyResult<f64> {
        self.inner.burning_cost(&historical_losses.0, attachment.0, limit.0, exposure_years.0).py_err()
    }

    #[pyo3(signature = (deductible, expected_loss, severity_cv))]
    fn loss_elimination_ratio(&self, deductible: CFloat, expected_loss: CFloat, severity_cv: CFloat) -> PyResult<f64> {
        self.inner.loss_elimination_ratio(deductible.0, expected_loss.0, severity_cv.0).py_err()
    }

    #[pyo3(signature = (layer_std_dev, risk_load_multiple))]
    fn excess_risk_load(&self, layer_std_dev: CFloat, risk_load_multiple: CFloat) -> PyResult<f64> {
        self.inner.excess_risk_load(layer_std_dev.0, risk_load_multiple.0).py_err()
    }
}

/// Rate-level indications: on-levelling, trend, combined and permissible loss ratios.
#[pyclass(module = "riskpy.cpp_underwriter", name = "RateAnalyzer", subclass, weakref, frozen)]
pub struct RateAnalyzer {
    inner: core::rate::RateAnalyzer,
}

#[pymethods]
impl RateAnalyzer {
    #[new]
    #[pyo3(signature = (target_combined_ratio = CFloat(1.0), expense_ratio = CFloat(0.3)), text_signature = "(target_combined_ratio=1.0, expense_ratio=0.3)")]
    fn new(target_combined_ratio: CFloat, expense_ratio: CFloat) -> PyResult<Self> {
        Ok(RateAnalyzer { inner: core::rate::RateAnalyzer::new(target_combined_ratio.0, expense_ratio.0).py_err()? })
    }

    #[pyo3(signature = (rate_changes))]
    fn on_level_factor(&self, rate_changes: CFloats) -> PyResult<f64> {
        self.inner.on_level_factor(&rate_changes.0).py_err()
    }

    #[pyo3(signature = (old_premium, new_premium))]
    fn rate_change_impact(&self, old_premium: CFloat, new_premium: CFloat) -> PyResult<f64> {
        self.inner.rate_change_impact(old_premium.0, new_premium.0).py_err()
    }

    #[pyo3(signature = (loss_ratio))]
    fn combined_ratio(&self, loss_ratio: CFloat) -> PyResult<f64> {
        self.inner.combined_ratio(loss_ratio.0).py_err()
    }

    #[pyo3(signature = (current_loss_ratio))]
    fn required_rate_change(&self, current_loss_ratio: CFloat) -> PyResult<f64> {
        self.inner.required_rate_change(current_loss_ratio.0).py_err()
    }

    #[pyo3(signature = (earned_premiums, rate_changes))]
    fn on_level_premiums(&self, earned_premiums: CFloats, rate_changes: CFloats) -> PyResult<Vec<f64>> {
        self.inner.on_level_premiums(&earned_premiums.0, &rate_changes.0).py_err()
    }

    fn permissible_loss_ratio(&self) -> f64 {
        self.inner.permissible_loss_ratio()
    }

    #[pyo3(signature = (annual_trend, years))]
    fn trend_factor(&self, annual_trend: CFloat, years: CFloat) -> PyResult<f64> {
        self.inner.trend_factor(annual_trend.0, years.0).py_err()
    }

    #[pyo3(signature = (incurred_losses, earned_exposure))]
    fn loss_cost_rate(&self, incurred_losses: CFloat, earned_exposure: CFloat) -> PyResult<f64> {
        self.inner.loss_cost_rate(incurred_losses.0, earned_exposure.0).py_err()
    }
}

// =====================================================================================
// FourierTransform

/// Radix-2 FFT, convolution and the compound-Poisson aggregate PMF (static methods).
#[pyclass(module = "riskpy.cpp_underwriter", name = "FourierTransform", subclass, weakref, frozen)]
pub struct FourierTransform;

#[pymethods]
impl FourierTransform {
    /// Forward DFT of (real, imag), zero-padded to the next power of two.
    #[staticmethod]
    #[pyo3(signature = (real, imag))]
    fn fft(py: Python<'_>, real: CFloats, imag: CFloats) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.detach(|| core::fourier::fft(&real.0, &imag.0)).py_err()
    }

    /// Inverse DFT (normalised by 1/N), zero-padded to the next power of two.
    #[staticmethod]
    #[pyo3(signature = (real, imag))]
    fn ifft(py: Python<'_>, real: CFloats, imag: CFloats) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.detach(|| core::fourier::ifft(&real.0, &imag.0)).py_err()
    }

    /// Linear convolution, length len(a) + len(b) - 1 (empty if either is empty).
    #[staticmethod]
    #[pyo3(signature = (a, b))]
    fn convolve(py: Python<'_>, a: CFloats, b: CFloats) -> PyResult<Vec<f64>> {
        py.detach(|| core::fourier::convolve(&a.0, &b.0)).py_err()
    }

    /// Compound-Poisson aggregate PMF on the severity's grid; grid_size is a power of
    /// two up to 2**24, or 0 to size the grid from the frequency and the severity.
    #[staticmethod]
    #[pyo3(signature = (severity_pmf, expected_frequency, grid_size = CInt(0)),
           text_signature = "(severity_pmf, expected_frequency, grid_size=0)")]
    fn compound_poisson_pmf(py: Python<'_>, severity_pmf: CFloats, expected_frequency: CFloat, grid_size: CInt) -> PyResult<Vec<f64>> {
        py.detach(|| core::fourier::compound_poisson_pmf(&severity_pmf.0, expected_frequency.0, grid_size.0)).py_err()
    }
}

/// RiskPY — High Performance Actuarial Engine
#[pymodule]
fn cpp_underwriter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Field>()?;
    m.add_class::<ExcelExporter>()?;
    m.add_class::<RiskEngine>()?;
    m.add_class::<ActuarialMath>()?;
    m.add_class::<FactorModel>()?;
    m.add_class::<MonteCarloSimulator>()?;
    m.add_class::<LossTriangle>()?;
    m.add_class::<ExperienceRating>()?;
    m.add_class::<ExposureRating>()?;
    m.add_class::<RateAnalyzer>()?;
    m.add_class::<FourierTransform>()?;
    // riskpy-doctor compares this with the Python package's version.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
