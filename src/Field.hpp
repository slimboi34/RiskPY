#pragma once
#include <string>

struct Field {
  std::string name;
  std::string label;
  std::string type;

  Field() = default;
  Field(std::string n, std::string l, std::string t)
      : name(std::move(n)), label(std::move(l)), type(std::move(t)) {}
};
