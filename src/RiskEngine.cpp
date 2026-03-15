#include "RiskEngine.hpp"
#include <iostream>

void RiskEngine::add_field(const Field &field) { fields.push_back(field); }

std::vector<Field> RiskEngine::get_fields() const { return fields; }

void RiskEngine::set_math_logic(std::function<double(const std::map<std::string, DynamicField>&)> logic) {
  math_logic = logic;
}

void RiskEngine::attach_exporter(ExcelExporter *exporter) {
  current_exporter = exporter;
}

double RiskEngine::execute(const std::map<std::string, DynamicField>& inputs) {
  last_inputs = inputs;

  if (math_logic) {
    last_premium = math_logic(inputs);
    return last_premium;
  }
  return 0.0;
}

void RiskEngine::export_to_excel(const std::string &filename) {
  if (current_exporter) {
    current_exporter->write_data(filename, last_inputs, last_premium);
  } else {
    std::cerr << "No exporter attached!" << std::endl;
  }
}
