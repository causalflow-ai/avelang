# Python bindings for ave-lang

# Include all targets for the LLVMInitialize*Target symbols in the bindings.
llvm_map_components_to_libnames(llvm_libs support core irreader ${LLVM_TARGETS_TO_BUILD})

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
    ${mlir_libs}
)

# Set output properties
set_target_properties(_avelang_bindings PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Set output directory if specified by setup.py
if(DEFINED AVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_avelang_bindings PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${AVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY}"
    )
endif()
