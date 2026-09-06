add_library(ave-lang-target-gpu
  lower_to_llvm.cc
  reconcile_nvgpu_index_bitwidth.cc
  gpu_outlining.cc
  gpu_backend.cc
  link_intrinsic_implementation.cc
)

target_link_libraries(ave-lang-target-gpu
  PRIVATE
  LLVMCore
  LLVMSupport
  MLIRLLVMDialect
  MLIRTargetLLVMIRExport
  MLIRArithToLLVM
  MLIRComplexToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRGPUDialect
  MLIRFuncAllExtensions
  MLIRFuncInlinerExtension
  MLIRVectorToLLVM
  MLIRUBToLLVM
  MLIRNVVMToLLVM
  MLIRAffineDialect
  MLIRPass
  ave-lang-mlir
  ave-lang-dialect-transforms
)

if(WITH_AMDGPU)
  target_link_libraries(ave-lang-target-gpu PRIVATE
    ave-lang-target-amdgpu
  )
endif()

if(WITH_CUDA)
  target_link_libraries(ave-lang-target-gpu PRIVATE
    ave-lang-target-nvvm
  )
endif()
target_compile_definitions(ave-lang-target-gpu PRIVATE ${LLVM_DEFINITIONS})

add_executable(lower_to_llvm_test lower_to_llvm_test.cc)
target_link_libraries(lower_to_llvm_test 
  ave-lang-target-gpu
  GTest::gtest_main
  MLIRParser
)
add_test(lower_to_llvm_test lower_to_llvm_test)

add_executable(gpu_outlining_test gpu_outlining_test.cc)
target_link_libraries(gpu_outlining_test 
  ave-lang-target-gpu
  GTest::gtest_main
  MLIRParser
)
add_test(gpu_outlining_test gpu_outlining_test)
