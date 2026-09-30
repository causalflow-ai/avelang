cmake_minimum_required(VERSION 3.25)
cmake_policy(VERSION 3.25)
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
set(AVE_LANG_BACKEND "cuda" CACHE STRING "GPU backends to use: cuda, rocm, or both (semicolon-separated list)")

# Parse the backend list and set appropriate flags
set(WITH_CUDA OFF)
set(WITH_AMDGPU OFF)

# Convert string to list if needed
if(AVE_LANG_BACKEND)
    foreach(backend IN LISTS AVE_LANG_BACKEND)
        string(STRIP "${backend}" backend)
        string(TOLOWER "${backend}" backend)

        if(backend STREQUAL "cuda")
            set(WITH_CUDA ON)
            message(STATUS "CUDA backend enabled")
        elseif(backend STREQUAL "rocm")
            set(WITH_AMDGPU ON)
            message(STATUS "ROCm backend enabled")
        elseif(NOT backend STREQUAL "")
            message(FATAL_ERROR "Invalid backend '${backend}'. Valid backends are: cuda, rocm")
        endif()
    endforeach()
endif()

# Ensure at least one backend is enabled
if(NOT WITH_CUDA AND NOT WITH_AMDGPU)
    message(FATAL_ERROR "At least one backend must be enabled. Valid backends are: cuda, rocm")
endif()

option(WITH_PYTHON "Enable Python bindings" OFF)

set(HUNTER_PACKAGES pybind11 GTest ZLIB zstd)
set(HUNTER_FILEPATH_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hunter_config.cmake")

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")

FetchContent_Declare(
    SetupHunter
    GIT_REPOSITORY https://github.com/cpp-pm/gate
    GIT_TAG v0.11.0
)
FetchContent_MakeAvailable(SetupHunter)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(ave-lang
    VERSION 0.1.0
    DESCRIPTION "Ave DSL Compiler"
    LANGUAGES CXX
)

add_compile_options(-Wall -fvisibility=hidden -fPIC)
set(CMAKE_CXX_FLAGS_DEBUG "-g")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)
find_package(LLVM 23.0 CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)
find_package(Clang REQUIRED CONFIG)

# Locate build tool executables using paths exported by LLVM/MLIR packages.
if(NOT MLIR_OPT_EXECUTABLE)
    set(_mlir_opt_hints)
    if(DEFINED MLIR_TOOLS_DIR)
        list(APPEND _mlir_opt_hints "${MLIR_TOOLS_DIR}")
    endif()
    if(DEFINED LLVM_TOOLS_BINARY_DIR)
        list(APPEND _mlir_opt_hints "${LLVM_TOOLS_BINARY_DIR}")
    endif()
    find_program(MLIR_OPT_EXECUTABLE NAMES mlir-opt
                 HINTS ${_mlir_opt_hints}
                 REQUIRED)
endif()
set(MLIR_OPT_EXECUTABLE "${MLIR_OPT_EXECUTABLE}" CACHE FILEPATH "Path to the mlir-opt executable")

if(NOT LLVM_OBJCOPY_EXECUTABLE)
    if(DEFINED LLVM_OBJCOPY AND EXISTS "${LLVM_OBJCOPY}")
        set(LLVM_OBJCOPY_EXECUTABLE "${LLVM_OBJCOPY}")
    else()
        set(_llvm_objcopy_hints)
        if(DEFINED LLVM_TOOLS_BINARY_DIR)
            list(APPEND _llvm_objcopy_hints "${LLVM_TOOLS_BINARY_DIR}")
        endif()
        find_program(LLVM_OBJCOPY_EXECUTABLE NAMES llvm-objcopy
                     HINTS ${_llvm_objcopy_hints}
                     REQUIRED)
    endif()
endif()
set(LLVM_OBJCOPY_EXECUTABLE "${LLVM_OBJCOPY_EXECUTABLE}" CACHE FILEPATH "Path to the llvm-objcopy executable")

# Include MLIR's cmake modules for tablegen support
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)
find_package(GTest REQUIRED CONFIG)

# Add -Werror only when not building Python bindings (to avoid warnings from external code)
if(NOT WITH_PYTHON)
    add_compile_options(-Werror)
endif()

if(WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA support enabled")
else()
  message(STATUS "CUDA support disabled")
endif()

if(WITH_AMDGPU)
  message(STATUS "AMDGPU support enabled")
  find_package(hip REQUIRED)

  # Extract ROCm paths from HIP package detection
  set(ROCM_ROOT_PATH "${HIP_PACKAGE_PREFIX_DIR}")
  set(ROCM_DEVICE_LIBRARY_PATH "${ROCM_ROOT_PATH}/amdgcn/bitcode")
  set(ROCM_TOOL_PATHS "${ROCM_ROOT_PATH}/bin;${ROCM_ROOT_PATH}/llvm/bin")

  message(STATUS "ROCm root path: ${ROCM_ROOT_PATH}")
  message(STATUS "ROCm device library path: ${ROCM_DEVICE_LIBRARY_PATH}")
  message(STATUS "ROCm tool paths: ${ROCM_TOOL_PATHS}")
else()
  message(STATUS "AMDGPU support disabled")
endif()

include_directories(include lib)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/avelang)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/avelang/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/avelang/config.h
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

enable_testing()

add_subdirectory(lib)

# Only build Python bindings if requested
if(WITH_PYTHON)
    add_subdirectory(python)
endif()
