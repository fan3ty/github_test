cmake_minimum_required(VERSION 3.13)
project(test_dealii)

find_package(deal.II 9.5.0 REQUIRED)

message(STATUS "deal.II version: ${DEAL_II_VERSION}")
message(STATUS "deal.II features: ${DEAL_II_FEATURES}")
message(STATUS "deal.II with MPI: ${DEAL_II_WITH_MPI}")
message(STATUS "deal.II with p4est: ${DEAL_II_WITH_P4EST}")
message(STATUS "deal.II with Trilinos: ${DEAL_II_WITH_TRILINOS}")
message(STATUS "deal.II with SUNDIALS: ${DEAL_II_WITH_SUNDIALS}")
