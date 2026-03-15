# Install script for directory: /Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX/headers" TYPE FILE FILES "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-build/OpenXLSX/OpenXLSX-Exports.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX/headers" TYPE FILE FILES
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/IZipArchive.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLCell.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLCellIterator.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLCellRange.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLCellReference.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLCellValue.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLColor.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLColumn.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLCommandQuery.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLComments.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLConstants.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLContentTypes.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLDateTime.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLDocument.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLDrawing.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLException.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLFormula.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLIterator.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLMergeCells.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLProperties.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLRelationships.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLRow.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLRowData.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLSharedStrings.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLSheet.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLStyles.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLTables.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLWorkbook.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLXmlData.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLXmlFile.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLXmlParser.hpp"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/headers/XLZipArchive.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/OpenXLSX" TYPE FILE FILES "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/OpenXLSX.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "lib" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/josh/Projects/underwriting_python_framework/build/output/libOpenXLSX.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenXLSX.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenXLSX.a")
    execute_process(COMMAND "/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenXLSX.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-src/OpenXLSX/OpenXLSXConfig.cmake"
    "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-build/OpenXLSX/OpenXLSX/OpenXLSXConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake"
         "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-build/OpenXLSX/CMakeFiles/Export/c72cc94553a1a0c9b05f75dae42fb1d7/OpenXLSXTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX/OpenXLSXTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-build/OpenXLSX/CMakeFiles/Export/c72cc94553a1a0c9b05f75dae42fb1d7/OpenXLSXTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenXLSX" TYPE FILE FILES "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-build/OpenXLSX/CMakeFiles/Export/c72cc94553a1a0c9b05f75dae42fb1d7/OpenXLSXTargets-noconfig.cmake")
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/josh/Projects/underwriting_python_framework/build/_deps/openxlsx-build/OpenXLSX/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
