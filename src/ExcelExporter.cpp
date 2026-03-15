#include "ExcelExporter.hpp"
#include <OpenXLSX.hpp>
#include <iostream>

using namespace OpenXLSX;

ExcelExporter::ExcelExporter(const std::string &template_name)
    : template_name(template_name) {}

void ExcelExporter::map_column(const std::string &col, const std::string &field_name,
                               const std::string &label) {
  column_mapping[col] = {field_name, label};
}

void ExcelExporter::write_data(const std::string &filename, const std::map<std::string, DynamicField>& inputs,
                               double premium) {
  XLDocument doc;
  doc.create(filename);
  auto wks = doc.workbook().worksheet("Sheet1");

  // Write Headers based on mapping
  for (const auto &[col, data] : column_mapping) {
    const auto& [field_name, label] = data;
    wks.cell(col + "1").value() = label;
    
    if (field_name == "premium_output") {
      wks.cell(col + "2").value() = premium;
    } else {
      if (inputs.find(field_name) != inputs.end()) {
        const auto& val = inputs.at(field_name);
        if (std::holds_alternative<double>(val)) {
            wks.cell(col + "2").value() = std::get<double>(val);
        } else if (std::holds_alternative<std::string>(val)) {
            wks.cell(col + "2").value() = std::get<std::string>(val);
        }
      }
    }
  }

  doc.save();
  doc.close();
  std::cout << "Successfully exported data to " << filename << std::endl;
}
