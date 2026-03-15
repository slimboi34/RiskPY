#pragma once
#include <map>
#include <string>
#include <utility>
#include <variant>

using DynamicField = std::variant<double, std::string>;

class ExcelExporter {
public:
  ExcelExporter(const std::string &template_name);
  void map_column(const std::string &col, const std::string &field_name, const std::string &label);
  void write_data(const std::string &filename, const std::map<std::string, DynamicField>& inputs,
                  double premium);

private:
  std::string template_name;
  std::map<std::string, std::pair<std::string, std::string>> column_mapping;
};
