cmake_minimum_required(VERSION 3.24)

project(PendulumSolver LANGUAGES CXX CUDA)

# C++ / CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation (-rdc=true)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Release optimization (-O3)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# MSVC preprocessor option (/Zc:preprocessor)
add_compile_options(
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:preprocessor>
    $<$<COMPILE_LANG_AND_ID:CUDA,MSVC>:-Xcompiler=/Zc:preprocessor>
)

# Source files
add_executable(PendulumSolver
    PendulumSolverCUDA.cu
    PendulumFractals.cu
    ODE.cu
    DiffEqSolverCUDA.cu
)

# Project includes
target_include_directories(PendulumSolver PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Include"
    "C:/vcpkg/installed/x64-windows-static/include"
)

# Library search path
target_link_directories(PendulumSolver PRIVATE
    "C:/vcpkg/installed/x64-windows-static/lib"
)

# Libraries
target_link_libraries(PendulumSolver PRIVATE
    pngwriter
    png16
    z
    bz2
    brotlidec
    brotlienc
    freetype
    brotlicommon
)

# Use MSVC's cl.exe as CUDA host compiler
# if(MSVC)
    set_property(TARGET PendulumSolver PROPERTY
        CUDA_HOST_COMPILER
        "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe"
    )
# endif()
