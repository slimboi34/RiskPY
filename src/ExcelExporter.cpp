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

void ExcelExporter::write_batch_data(const std::string &filename, const std::vector<std::map<std::string, DynamicField>>& batch_inputs,
                                     const std::vector<double>& batch_premiums) {
  XLDocument doc;
  doc.create(filename);
  auto wks = doc.workbook().worksheet("Sheet1");

  // Write Headers based on mapping
  for (const auto &[col, data] : column_mapping) {
    const auto& [field_name, label] = data;
    wks.cell(col + "1").value() = label;
  }

  // Write rows
  for (size_t i = 0; i < batch_inputs.size(); ++i) {
      std::string row_str = std::to_string(i + 2); // Start at row 2
      const auto& inputs = batch_inputs[i];
      double premium = (i < batch_premiums.size()) ? batch_premiums[i] : 0.0;
      
      for (const auto &[col, data] : column_mapping) {
        const auto& [field_name, label] = data;
        
        if (field_name == "premium_output") {
          wks.cell(col + row_str).value() = premium;
        } else {
          if (inputs.find(field_name) != inputs.end()) {
            const auto& val = inputs.at(field_name);
            if (std::holds_alternative<double>(val)) {
                wks.cell(col + row_str).value() = std::get<double>(val);
            } else if (std::holds_alternative<std::string>(val)) {
                wks.cell(col + row_str).value() = std::get<std::string>(val);
            }
          }
        }
      }
  }

  doc.save();
  doc.close();
  std::cout << "Successfully exported batch data to " << filename << std::endl;
}
