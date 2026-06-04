cmake_minimum_required(VERSION 3.14)
project(hamiltonian_pybind)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(_hamiltonian hamiltonian_pybind.cpp)
target_link_libraries(_hamiltonian PRIVATE Eigen3::Eigen)
target_include_directories(_hamiltonian PRIVATE /usr/include/eigen3)

# Install into the project directory
install(TARGETS _hamiltonian
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR})