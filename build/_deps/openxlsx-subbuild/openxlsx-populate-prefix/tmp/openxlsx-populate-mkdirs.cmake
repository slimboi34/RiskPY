# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src")
  file(MAKE_DIRECTORY "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src")
endif()
file(MAKE_DIRECTORY
  "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-build"
  "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-subbuild/openxlsx-populate-prefix"
  "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-subbuild/openxlsx-populate-prefix/tmp"
  "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-subbuild/openxlsx-populate-prefix/src/openxlsx-populate-stamp"
  "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-subbuild/openxlsx-populate-prefix/src"
  "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-subbuild/openxlsx-populate-prefix/src/openxlsx-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-subbuild/openxlsx-populate-prefix/src/openxlsx-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-subbuild/openxlsx-populate-prefix/src/openxlsx-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
