cmake_minimum_required(VERSION 3.23)
project(JunqiEnv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED Python_EXECUTABLE AND DEFINED ENV{CONDA_PREFIX})
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/python.exe" CACHE FILEPATH "Python interpreter")
endif()

if(NOT DEFINED Python_ROOT_DIR AND DEFINED ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Python root directory")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT EXISTS "${pybind11_DIR}/pybind11Config.cmake" AND Python_EXECUTABLE)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_result
    )
    if(_pybind11_result EQUAL 0 AND EXISTS "${_pybind11_cmake_dir}")
        set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 CMake package directory" FORCE)
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

set(JUNQI_SOURCES
    junqi_types.h
    junqi_board.h
    junqi_board.cpp
    junqi_state.h
    junqi_state.cpp
    setup_validator.h
    setup_validator.cpp
    move_generator.h
    move_generator.cpp
    combat_resolver.h
    combat_resolver.cpp
    setup_generator.h
    setup_generator.cpp
    junqi_environment.h
    junqi_environment.cpp
    junqi_batch_environment.h
    junqi_batch_environment.cpp
    junqi_bindings.cpp
)

pybind11_add_module(JunqiEnv ${JUNQI_SOURCES})

if(NOT JUNQI_OUTPUT_DIR)
    set(JUNQI_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/../build")
endif()

set_target_properties(JunqiEnv PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${JUNQI_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${JUNQI_OUTPUT_DIR}"
)

if(MSVC)
    target_compile_options(JunqiEnv PRIVATE /O2 /permissive-)
else()
    target_compile_options(JunqiEnv PRIVATE -O3 -Wall -Wextra)
    target_link_options(JunqiEnv PRIVATE -static)
endif()
