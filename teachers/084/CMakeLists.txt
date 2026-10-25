cmake_minimum_required(VERSION 3.23)
project(fast_rule_ab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
set(pybind11_DIR "${_pybind11_dir}" CACHE PATH "" FORCE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fast_rule_ab fast_rule_ab.cpp)
if(MSVC)
    target_compile_options(fast_rule_ab PRIVATE /O2 /permissive-)
else()
    target_compile_options(fast_rule_ab PRIVATE -O3 -Wall -Wextra)
endif()
set_target_properties(fast_rule_ab PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

