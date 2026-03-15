#include "ActuarialMath.hpp"
#include "FactorModel.hpp"
#include "MonteCarlo.hpp"
#include "ExcelExporter.hpp"
#include "Field.hpp"
#include "RiskEngine.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_underwriter, m) {
  m.doc() = "C++ Underwriting Framework Core Engine";

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
      .def("export_to_excel", &RiskEngine::export_to_excel);

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
}
