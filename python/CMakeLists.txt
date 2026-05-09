# Python bindings for ave-lang

# Find required LLVM/MLIR libraries
# Use all-targets to get all LLVM target libraries (needed for MLIR)
llvm_map_components_to_libnames(llvm_libs support core irreader)
# Get all target libraries via llvm-config to ensure all LLVMInitialize*Target symbols are available
execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs all-targets
    OUTPUT_VARIABLE LLVM_ALL_TARGETS_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Convert the space-separated list to a CMake list
separate_arguments(LLVM_ALL_TARGETS_LIBS NATIVE_COMMAND "${LLVM_ALL_TARGETS_LIBS}")

# Get MLIR libraries
get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)

# Create Python extension module
pybind11_add_module(_avelang_bindings src/avelang_bindings.cc)

# Set target properties
target_compile_definitions(_avelang_bindings PRIVATE 
    VERSION_INFO=\"${PROJECT_VERSION}\"
)

# Filter out problematic compiler flags for GCC compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(_avelang_bindings PRIVATE
        -Wno-unknown-warning-option
        -Wno-unused-command-line-argument
    )
endif()

# Link against avelang libraries and LLVM/MLIR
target_link_directories(_avelang_bindings PRIVATE ${LLVM_LIBRARY_DIRS})

target_link_libraries(_avelang_bindings PRIVATE
    ave-lang-driver
    ave-lang-basic
    ave-lang-mlir
    ave-lang-frontend
    ave-lang-ast
    clangFrontend
    ${llvm_libs}
    ${LLVM_ALL_TARGETS_LIBS}
    ${mlir_libs}
)

# Set output properties
set_target_properties(_avelang_bindings PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Keep the in-tree Python package usable for development builds. setup.py
# supplies AVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY for wheel builds.
if(NOT DEFINED AVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY)
    set(AVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set_target_properties(_avelang_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${AVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY}"
)
