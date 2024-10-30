set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake)
cmake_minimum_required(VERSION 3.21)
project(Mumps_mex CXX)

include($ENV{CMAKE_INSTALL_PREFIX}/cmake/MexUtilities.cmake)
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/GKlibTargets.cmake)
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/metisTargets.cmake)
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/mumpsTargets.cmake)

set(MKL_LINK static)
set(MKL_THREADING sequential)
set(MKL_INTERFACE lp64)
find_package(MKL REQUIRED)

find_package(Matlab REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

matlab_add_mex(
    NAME mumps_mex
    SRC src/mumps.cpp
    LINK_TO MexUtilities::MexUtilities mumps::libdmumps
    R2018a
)
