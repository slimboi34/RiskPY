#include "ActuarialMath.hpp"
#include "ExperienceRating.hpp"
#include "ExposureRating.hpp"
#include "FactorModel.hpp"
#include "LossTriangle.hpp"
#include "MonteCarlo.hpp"
#include "RateAnalyzer.hpp"
#include "ExcelExporter.hpp"
#include "Field.hpp"
#include "RiskEngine.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_underwriter, m) {
  m.doc() = "RiskPY — High Performance Actuarial Engine";

  py::class_<Field>(m, "Field")
      .def(py::init<std::string, std::string, std::string>(),
           py::arg("name") = "", py::arg("label") = "", py::arg("type") = "")
      .def_readwrite("name", &Field::name)
      .def_readwrite("label", &Field::label)
      .def_readwrite("type", &Field::type);

  py::class_<ExcelExporter>(m, "ExcelExporter")
      .def(py::init<const std::string &>(), py::arg("template"))
      .def("map_column", &ExcelExporter::map_column);

  py::class_<RiskEngine>(m, "RiskEngine")
      .def(py::init<>())
      .def("add_field", &RiskEngine::add_field)
      .def("get_fields", &RiskEngine::get_fields)
      .def("set_math_logic", &RiskEngine::set_math_logic)
      .def("attach_exporter", &RiskEngine::attach_exporter)
      .def("execute", &RiskEngine::execute)
      .def("export_to_excel", &RiskEngine::export_to_excel)
      .def("export_batch_to_excel", &RiskEngine::export_batch_to_excel);

  py::class_<ActuarialMath>(m, "ActuarialMath")
      .def_static("present_value", &ActuarialMath::present_value, py::arg("rate"), py::arg("periods"), py::arg("payment"))
      .def_static("future_value", &ActuarialMath::future_value, py::arg("rate"), py::arg("periods"), py::arg("payment"))
      .def_static("calculate_loss_ratio", &ActuarialMath::calculate_loss_ratio, py::arg("incurred_losses"), py::arg("earned_premium"))
      .def_static("lookup_mortality_rate", &ActuarialMath::lookup_mortality_rate, py::arg("age"));

  py::class_<FactorModel>(m, "FactorModel")
      .def(py::init<double>(), py::arg("initial_base_rate") = 0.0)
      .def("set_base_rate", &FactorModel::set_base_rate, py::arg("rate"))
      .def("add_multiplier", &FactorModel::add_multiplier, py::arg("field_name"), py::arg("exact_value"), py::arg("multiplier"))
      .def("add_numeric_band_multiplier", &FactorModel::add_numeric_band_multiplier, py::arg("field_name"), py::arg("min_val"), py::arg("max_val"), py::arg("multiplier"))
      .def("calculate", &FactorModel::calculate, py::arg("inputs"));

  py::class_<MonteCarloSimulator>(m, "MonteCarloSimulator")
      .def_static("simulate_aggregate_loss", &MonteCarloSimulator::simulate_aggregate_loss, py::arg("trials"), py::arg("expected_frequency"), py::arg("expected_severity_mu"), py::arg("severity_sigma"))
      .def_static("simulate_life_portfolio", &MonteCarloSimulator::simulate_life_portfolio, py::arg("trials"), py::arg("policy_count"), py::arg("base_mortality_rate"), py::arg("shock_volatility"), py::arg("death_benefit"));

  // ===== NEW ENTERPRISE MODULES =====

  py::class_<LossTriangle>(m, "LossTriangle")
      .def(py::init<>())
      .def("add_origin_year", &LossTriangle::add_origin_year, py::arg("year"), py::arg("cumulative_payments"))
      .def("get_development_factors", &LossTriangle::get_development_factors)
      .def("get_ultimate_losses", &LossTriangle::get_ultimate_losses)
      .def("get_ibnr_reserves", &LossTriangle::get_ibnr_reserves)
      .def("get_origin_year_count", &LossTriangle::get_origin_year_count)
      .def("get_development_period_count", &LossTriangle::get_development_period_count);

  py::class_<ExperienceRating>(m, "ExperienceRating")
      .def_static("credibility_weighted_rate", &ExperienceRating::credibility_weighted_rate, py::arg("observed_rate"), py::arg("expected_rate"), py::arg("credibility_z"))
      .def_static("calculate_credibility", &ExperienceRating::calculate_credibility, py::arg("claim_count"), py::arg("k_parameter"))
      .def_static("experience_mod_factor", &ExperienceRating::experience_mod_factor, py::arg("actual_losses"), py::arg("expected_losses"), py::arg("ballast"))
      .def_static("buhlmann_straub_credibility", &ExperienceRating::buhlmann_straub_credibility, py::arg("exposure"), py::arg("variance_ratio"))
      .def_static("expected_losses", &ExperienceRating::expected_losses, py::arg("exposure"), py::arg("frequency"), py::arg("severity"));

  py::class_<ExposureRating>(m, "ExposureRating")
      .def_static("increased_limits_factor", &ExposureRating::increased_limits_factor, py::arg("base_limit"), py::arg("target_limit"), py::arg("b_parameter"))
      .def_static("layer_premium", &ExposureRating::layer_premium, py::arg("ground_up_premium"), py::arg("attachment"), py::arg("limit"), py::arg("ilf_at_base"), py::arg("ilf_at_attachment"), py::arg("ilf_at_limit"))
      .def_static("burning_cost", &ExposureRating::burning_cost, py::arg("historical_losses"), py::arg("attachment"), py::arg("limit"), py::arg("exposure_years"))
      .def_static("loss_elimination_ratio", &ExposureRating::loss_elimination_ratio, py::arg("deductible"), py::arg("expected_loss"), py::arg("severity_cv"))
      .def_static("excess_risk_load", &ExposureRating::excess_risk_load, py::arg("layer_std_dev"), py::arg("risk_load_multiple"));

  py::class_<RateAnalyzer>(m, "RateAnalyzer")
      .def_static("on_level_factor", &RateAnalyzer::on_level_factor, py::arg("rate_changes"))
      .def_static("rate_change_impact", &RateAnalyzer::rate_change_impact, py::arg("old_premium"), py::arg("new_premium"))
      .def_static("combined_ratio", &RateAnalyzer::combined_ratio, py::arg("loss_ratio"), py::arg("expense_ratio"))
      .def_static("required_rate_change", &RateAnalyzer::required_rate_change, py::arg("target_combined_ratio"), py::arg("current_loss_ratio"), py::arg("expense_ratio"))
      .def_static("on_level_premiums", &RateAnalyzer::on_level_premiums, py::arg("earned_premiums"), py::arg("rate_changes"))
      .def_static("permissible_loss_ratio", &RateAnalyzer::permissible_loss_ratio, py::arg("target_combined_ratio"), py::arg("expense_ratio"))
      .def_static("trend_factor", &RateAnalyzer::trend_factor, py::arg("annual_trend"), py::arg("years"))
      .def_static("loss_cost_rate", &RateAnalyzer::loss_cost_rate, py::arg("incurred_losses"), py::arg("earned_exposure"));
}
