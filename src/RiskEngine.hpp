#pragma once
#include "ExcelExporter.hpp"
#include "Field.hpp"
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <variant>

using DynamicField = std::variant<double, std::string>;

class RiskEngine {
public:
  RiskEngine() = default;

  void add_field(const Field &field);
  std::vector<Field> get_fields() const;

  void set_math_logic(std::function<double(const std::map<std::string, DynamicField>&)> logic);
  void attach_exporter(ExcelExporter *exporter);

  double execute(const std::map<std::string, DynamicField>& inputs);
  void export_to_excel(const std::string &filename);

private:
  std::vector<Field> fields;
  std::function<double(const std::map<std::string, DynamicField>&)> math_logic;
  ExcelExporter *current_exporter = nullptr;
  std::map<std::string, DynamicField> last_inputs;
  double last_premium = 0.0;
};
