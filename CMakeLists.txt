cmake_minimum_required(VERSION 3.15)
project(NightVision VERSION 0.1.0 LANGUAGES C CXX)

# Header-only library interface for embedded C/C++ targets
add_library(nightvision INTERFACE)

target_include_directories(nightvision INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(nightvision INTERFACE c_std_99 cxx_std_17)

# Installation specification
include(GNUInstallDirs)
install(TARGETS nightvision EXPORT NightVisionTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT NightVisionTargets
    FILE NightVisionConfig.cmake
    NAMESPACE NightVision::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NightVision
)
