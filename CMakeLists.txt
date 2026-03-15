cmake_minimum_required(VERSION 3.15)
project(cpp_underwriter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch Pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Fetch OpenXLSX
FetchContent_Declare(
  OpenXLSX
  GIT_REPOSITORY https://github.com/troldal/OpenXLSX.git
  GIT_TAG master
)
FetchContent_MakeAvailable(OpenXLSX)

pybind11_add_module(cpp_underwriter
    src/bindings.cpp
    src/RiskEngine.cpp
    src/ExcelExporter.cpp
    src/ActuarialMath.cpp
    src/FactorModel.cpp
    src/MonteCarlo.cpp
    src/LossTriangle.cpp
    src/ExperienceRating.cpp
    src/ExposureRating.cpp
    src/RateAnalyzer.cpp
)

target_link_libraries(cpp_underwriter PRIVATE OpenXLSX::OpenXLSX)

install(TARGETS cpp_underwriter DESTINATION riskpy)
