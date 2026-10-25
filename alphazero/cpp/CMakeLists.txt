cmake_minimum_required(VERSION 3.23)
project(JunqiAZ LANGUAGES CXX)

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
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Python root")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
if((NOT DEFINED pybind11_DIR OR NOT EXISTS "${pybind11_DIR}/pybind11Config.cmake")
   AND Python_EXECUTABLE)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_result)
    if(_pybind11_result EQUAL 0 AND EXISTS "${_pybind11_cmake_dir}")
        set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 package" FORCE)
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

# The distribution builds the historically isolated search module from source.
pybind11_add_module(JunqiAZRootExplore
    junqi_board.cpp junqi_state.cpp setup_validator.cpp move_generator.cpp
    combat_resolver.cpp setup_generator.cpp junqi_environment.cpp az_state.cpp
    root_explore/neural_puct.cpp root_explore/bindings.cpp)
target_include_directories(JunqiAZRootExplore PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

pybind11_add_module(JunqiAZ
    junqi_board.cpp
    junqi_state.cpp
    setup_validator.cpp
    move_generator.cpp
    combat_resolver.cpp
    setup_generator.cpp
    junqi_environment.cpp
    az_state.cpp
    neural_puct.cpp
    heuristic_084_evaluator.cpp
    junqi_az_bindings.cpp)

pybind11_add_module(JunqiOracle
    "${CMAKE_SOURCE_DIR}/../supplement/src/root_only_runtime.cpp"
    oracle_bindings.cpp)
target_include_directories(JunqiOracle PRIVATE
    "${CMAKE_SOURCE_DIR}/../supplement/src")

if(NOT JUNQI_AZ_OUTPUT_DIR)
    set(JUNQI_AZ_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/../build")
endif()
set_target_properties(JunqiAZ PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${JUNQI_AZ_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${JUNQI_AZ_OUTPUT_DIR}")
set_target_properties(JunqiOracle PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${JUNQI_AZ_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${JUNQI_AZ_OUTPUT_DIR}")
set_target_properties(JunqiAZRootExplore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${JUNQI_AZ_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${JUNQI_AZ_OUTPUT_DIR}")

if(MSVC)
    target_compile_options(JunqiAZRootExplore PRIVATE /O2 /permissive- /EHsc)
    target_compile_options(JunqiAZ PRIVATE /O2 /permissive- /EHsc)
    target_compile_options(JunqiOracle PRIVATE /O2 /permissive- /EHsc)
else()
    target_compile_options(JunqiAZRootExplore PRIVATE -O3 -Wall -Wextra)
    target_compile_options(JunqiAZ PRIVATE -O3 -Wall -Wextra)
    target_compile_options(JunqiOracle PRIVATE -O3 -Wall -Wextra)
    if(MINGW)
        target_link_options(JunqiAZRootExplore PRIVATE -static)
        # Avoid a runtime dependency on MSYS2's libwinpthread-1.dll.
        target_link_options(JunqiAZ PRIVATE -static)
    endif()
endif()
