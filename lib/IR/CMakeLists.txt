set(NVVM_INTRINSICS_TEXT ${CMAKE_CURRENT_SOURCE_DIR}/Intrinsics/nvvm_intrinsics.mlir)
set(AMDGPU_INTRINSICS_TEXT ${CMAKE_CURRENT_SOURCE_DIR}/Intrinsics/amdgpu_intrinsics.mlir)

set(NVVM_INTRINSICS_BC ${CMAKE_CURRENT_BINARY_DIR}/nvvm_intrinsics.mlirbc)
set(AMDGPU_INTRINSICS_BC ${CMAKE_CURRENT_BINARY_DIR}/amdgpu_intrinsics.mlirbc)

get_filename_component(NVVM_INTRINSICS_BC_NAME ${NVVM_INTRINSICS_BC} NAME)
get_filename_component(AMDGPU_INTRINSICS_BC_NAME ${AMDGPU_INTRINSICS_BC} NAME)

add_custom_command(
  OUTPUT ${NVVM_INTRINSICS_BC}
  COMMAND ${MLIR_OPT_EXECUTABLE} --emit-bytecode ${NVVM_INTRINSICS_TEXT} -o ${NVVM_INTRINSICS_BC}
  DEPENDS ${NVVM_INTRINSICS_TEXT}
  COMMENT "Compiling NVVM intrinsic library to bytecode"
)

add_custom_command(
  OUTPUT ${AMDGPU_INTRINSICS_BC}
  COMMAND ${MLIR_OPT_EXECUTABLE} --emit-bytecode ${AMDGPU_INTRINSICS_TEXT} -o ${AMDGPU_INTRINSICS_BC}
  DEPENDS ${AMDGPU_INTRINSICS_TEXT}
  COMMENT "Compiling AMDGPU intrinsic library to bytecode"
)

set(NVVM_INTRINSICS_OBJ ${CMAKE_CURRENT_BINARY_DIR}/nvvm_intrinsics_mlirbc.o)
set(AMDGPU_INTRINSICS_OBJ ${CMAKE_CURRENT_BINARY_DIR}/amdgpu_intrinsics_mlirbc.o)

get_filename_component(NVVM_INTRINSICS_OBJ_NAME ${NVVM_INTRINSICS_OBJ} NAME)
get_filename_component(AMDGPU_INTRINSICS_OBJ_NAME ${AMDGPU_INTRINSICS_OBJ} NAME)

add_custom_command(
  OUTPUT ${NVVM_INTRINSICS_OBJ}
  COMMAND ${LLVM_OBJCOPY_EXECUTABLE} --input-target=binary --output-target=elf64-x86-64 --binary-architecture=x86-64 --add-section .note.GNU-stack=/dev/null --set-section-flags .note.GNU-stack=readonly ${NVVM_INTRINSICS_BC_NAME} ${NVVM_INTRINSICS_OBJ_NAME}
  DEPENDS ${NVVM_INTRINSICS_BC}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Embedding NVVM intrinsic bytecode"
)

add_custom_command(
  OUTPUT ${AMDGPU_INTRINSICS_OBJ}
  COMMAND ${LLVM_OBJCOPY_EXECUTABLE} --input-target=binary --output-target=elf64-x86-64 --binary-architecture=x86-64 --add-section .note.GNU-stack=/dev/null --set-section-flags .note.GNU-stack=readonly ${AMDGPU_INTRINSICS_BC_NAME} ${AMDGPU_INTRINSICS_OBJ_NAME}
  DEPENDS ${AMDGPU_INTRINSICS_BC}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Embedding AMDGPU intrinsic bytecode"
)

set_source_files_properties(
  ${NVVM_INTRINSICS_OBJ} ${AMDGPU_INTRINSICS_OBJ}
  PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

add_library(ave-lang-mlir
  named_module.cc
  layout_operation.cc
  constant_folder.cc
  builtin_module.cc
  type_system.cc
  Intrinsics/intrinsic_support.cc
  Intrinsics/nvvm_module.cc
  Intrinsics/amdgpu_module.cc
  mangler.cc
  mlir_generator.cc
  expr_generator.cc
  function_generator.cc
  symbol_table.cc
  symbol_scope.cc
  type_promotion.cc
  parsing_utils.cc
  ir_context.cc
  ${NVVM_INTRINSICS_OBJ}
  ${AMDGPU_INTRINSICS_OBJ}
)

add_dependencies(ave-lang-mlir ave-lang-dialect)

target_include_directories(ave-lang-mlir PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/lib/Dialect/AveLang/IR
    ${CMAKE_BINARY_DIR}/lib/Dialect/AveLang/IR
)

target_link_libraries(ave-lang-mlir
  ave-lang-basic
  ave-lang-utils
  ave-lang-dialect
  MLIRAffineDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRIndexDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRGPUDialect
  MLIRNVGPUDialect
  MLIRPtrDialect
  MLIRVectorDialect
  MLIRROCDLDialect
  MLIRNVVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRArithTransforms
  MLIRTensorTransforms
  MLIRBufferizationTransforms
  MLIRSCFTransforms
  MLIRControlFlowTransforms
)

add_executable(mlir_generator_test mlir_generator_test.cc)
target_link_libraries(mlir_generator_test ave-lang-frontend ave-lang-mlir ave-lang-dialect-transforms GTest::gtest_main)
add_test(mlir_generator_test mlir_generator_test)
