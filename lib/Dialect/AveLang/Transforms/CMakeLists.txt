add_library(ave-lang-dialect-transforms
  allocation_op_interface_impl.cc
  hoist_alloca_pass.cc
  normalize_ave_lang_return_pass.cc
  lower_ave_lang_to_memref_pass.cc
  lower_gpuop_to_intrinsics_pass.cc
)

target_include_directories(ave-lang-dialect-transforms PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/lib/Dialect/AveLang/IR
    ${CMAKE_BINARY_DIR}/lib/Dialect/AveLang/IR
)

target_link_libraries(ave-lang-dialect-transforms
  PUBLIC
    ave-lang-dialect
    ave-lang-mlir
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransforms
    MLIRFuncTransforms
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRVectorTransforms
    MLIRNVGPUDialect
    MLIRNVVMDialect
    MLIRROCDLDialect
    MLIRGPUDialect
    MLIRMemRefDialect
    MLIRPtrDialect
    MLIRBufferizationDialect
)
