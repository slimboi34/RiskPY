cmake_minimum_required(VERSION 3.15)
project(cpp_underwriter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Required on Linux/ELF when static deps (miniz/OpenXLSX) are linked into a .so
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  # Ensure FetchContent C targets (miniz) also emit PIC objects
  add_compile_options(-fPIC)
endif()

# ---------------------------------------------------------------------------
# Install speed & build stability
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Quiet, disconnected FetchContent: skip re-fetch on rebuilds; shallow clones below.
set(FETCHCONTENT_QUIET ON)
if(NOT DEFINED FETCHCONTENT_UPDATES_DISCONNECTED)
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

# ccache dramatically speeds up iterative rebuilds (dev + CI).
find_program(RISKPY_CCACHE ccache)
if(RISKPY_CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${RISKPY_CCACHE}")
  message(STATUS "RiskPY: using ccache (${RISKPY_CCACHE})")
endif()

if(MSVC)
  add_compile_options($<$<CONFIG:Release>:/O2>)
else()
  add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O2>
  )
endif()

include(FetchContent)

# Prefer an already-installed pybind11 (venv/system/cibuildwheel) to skip a git clone.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 2.11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Pin OpenXLSX to a release tag (not floating `master`) for reproducible installs.
# v0.5.1 is the last GitHub release before the project moved to Codeberg.
FetchContent_Declare(
  OpenXLSX
  GIT_REPOSITORY https://github.com/troldal/OpenXLSX.git
  GIT_TAG        v0.5.1
  GIT_SHALLOW    TRUE
)
# OpenXLSX pulls pugixml/zlib; keep its own tests/examples off for faster configure/build.
set(OPENXLSX_CREATE_DOCS OFF CACHE BOOL "" FORCE)
set(OPENXLSX_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(OPENXLSX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
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
    src/FourierMath.cpp
)

target_include_directories(cpp_underwriter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cpp_underwriter PRIVATE OpenXLSX::OpenXLSX)

install(TARGETS cpp_underwriter DESTINATION riskpy)
