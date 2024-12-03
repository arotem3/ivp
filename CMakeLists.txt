cmake_minimum_required(VERSION 3.10)

# Project name and version
project(ivp VERSION 1.0 LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create the interface library
add_library(ivp INTERFACE)

# Add include directories for the interface library
target_include_directories(ivp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Add the examples subdirectory
add_subdirectory(examples)

# Installation setup
include(GNUInstallDirs)

# Install the headers
install(FILES ivp.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY IVP DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the library target
install(TARGETS ivp EXPORT ivpTargets)

# Export the library for usage in other projects
install(EXPORT ivpTargets
    FILE ivpConfig.cmake
    NAMESPACE ivp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ivp
)
