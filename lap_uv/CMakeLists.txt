cmake_minimum_required(VERSION 3.18..3.22)
project(lap_uv)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(lap_uv src/main.cpp)
target_include_directories(lap_uv PRIVATE ./include)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(lap_uv PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

