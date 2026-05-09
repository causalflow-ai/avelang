#include "Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.h"
#include "IR/ir_context.h"
#include "avelang/config.h"
#include "gpu_passes.h"
#include "lower_to_llvm.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <gtest/gtest.h>

namespace causalflow::avelang::target::gpu {

class LLVMTargetTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ir_context_ = causalflow::avelang::ir::IRContext::Create();

#if defined(WITH_AMDGPU)
        static std::once_flag initAMDGPUFlag;
        std::call_once(initAMDGPUFlag, []() {
            LLVMInitializeAMDGPUTarget();
            LLVMInitializeAMDGPUTargetInfo();
            LLVMInitializeAMDGPUTargetMC();
            LLVMInitializeAMDGPUAsmPrinter();
        });
        compilation_options_.triple = "amdgcn-amd-amdhsa";
        compilation_options_.chipset = "gfx90a";
#elif defined(WITH_CUDA)
        static std::once_flag initNVPTXFlag;
        std::call_once(initNVPTXFlag, []() {
            LLVMInitializeNVPTXTarget();
            LLVMInitializeNVPTXTargetInfo();
            LLVMInitializeNVPTXTargetMC();
            LLVMInitializeNVPTXAsmPrinter();
        });
        compilation_options_.triple = "nvptx64-nvidia-cuda";
        compilation_options_.chipset = "sm_80";
#else
        GTEST_SKIP() << "No GPU backend available for LowerToLLVM tests";
#endif
    }

    mlir::OwningOpRef<mlir::ModuleOp>
    parseMLIRString(const std::string &mlirCode) {
        LowerToLLVM compiler(ir_context_.get());
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            mlirCode, compiler.getContext());
        EXPECT_TRUE(module);
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
        return module;
    }

    void TestLLVMCompilation(const std::string &mlirCode) {
        auto llvmModule = CompileToLLVM(mlirCode, nullptr, nullptr);
        ASSERT_NE(llvmModule, nullptr);
    }

    std::string LowerAveLangToMemRef(const std::string &mlirCode) {
        LowerToLLVM compiler(ir_context_.get());
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            mlirCode, compiler.getContext());
        EXPECT_TRUE(module);
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
        if (!module) {
            return {};
        }

        mlir::PassManager pm(compiler.getContext());
        pm.addPass(
            causalflow::avelang::dialect::createLowerAveLangToMemRefPass());
        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());
        EXPECT_TRUE(mlir::succeeded(pm.run(*module)));

        std::string lowered;
        llvm::raw_string_ostream os(lowered);
        module->print(os);
        return lowered;
    }

    std::unique_ptr<::llvm::Module> CompileToLLVM(const std::string &mlirCode,
                                                  std::string *mlirDump,
                                                  std::string *llvmDump) {
        LowerToLLVM compiler(ir_context_.get());
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            mlirCode, compiler.getContext());
        EXPECT_TRUE(module);
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
        if (!module) {
            return nullptr;
        }

        if (mlirDump) {
            llvm::raw_string_ostream mlirOs(*mlirDump);
            module->print(mlirOs);
        }

        auto llvmModule =
            compiler.compile(*module, llvmContext_, compilation_options_);

        EXPECT_NE(llvmModule, nullptr);
        if (!llvmModule) {
            return nullptr;
        }
        EXPECT_FALSE(llvmModule->empty());

        if (llvmDump) {
            llvm::raw_string_ostream llvmOs(*llvmDump);
            llvmModule->print(llvmOs, nullptr);
        }

        bool failed = llvm::verifyModule(*llvmModule, &llvm::errs());
        EXPECT_FALSE(failed);
        if (failed) {
            return nullptr;
        }

        return llvmModule;
    }

    std::unique_ptr<causalflow::avelang::ir::IRContext> ir_context_;
    ::llvm::LLVMContext llvmContext_;
    causalflow::avelang::target::gpu::GPUCompilationOptions
        compilation_options_;
};

TEST_F(LLVMTargetTest, SimpleFunction) {
    const std::string mlirCode = R"(
        module {
            func.func @test_func(%arg0: !ave.memref<!ave.layout<dims = [1], strides = []>, i32> {llvm.name = "out"}) attributes {ave.gpu_func = 2 : i32} {
                %c0 = arith.constant 0 : index
                %c42 = arith.constant 42 : i32
                ave.memref.store %c42, %arg0[%c0] : i32, !ave.memref<!ave.layout<dims = [1], strides = []>, i32>
                return
            }
        }
    )";
    TestLLVMCompilation(mlirCode);
}

TEST_F(LLVMTargetTest, FloorDiv) {
    const std::string mlirCode = R"(
        module {
          func.func @floor_div_test(%arg0: !ave.memref<!ave.layout<dims = [1], strides = []>, i32> {llvm.name = "out"}) attributes {ave.gpu_func = 2 : i32} {
            %c0 = arith.constant 0 : index
            %c10 = arith.constant 10 : i32
            %c3 = arith.constant 3 : i32
            %0 = arith.floordivsi %c10, %c3 : i32
            ave.memref.store %0, %arg0[%c0] : i32, !ave.memref<!ave.layout<dims = [1], strides = []>, i32>
            return
          }
        }
  )";
    TestLLVMCompilation(mlirCode);
}

TEST_F(LLVMTargetTest, MemrefStore) {
    const std::string mlirCode = R"(
module {
  func.func @matmul(%arg0: !ave.memref<!ave.layout<dims = [2, 3], strides = []>, i32> {llvm.name = "a"}) attributes {ave.gpu_func = 2 : i32} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = arith.index_cast %c1 : index to i32
    ave.memref.store %0, %arg0[%c0, %c0_0] : i32, !ave.memref<!ave.layout<dims = [2, 3], strides = []>, i32>
    return
  }
})";
    TestLLVMCompilation(mlirCode);
}

TEST_F(LLVMTargetTest, ForLoop) {
    const std::string mlirCode = R"(
module {
  func.func @for_loop_test(%arg0: !ave.memref<!ave.layout<dims = [2, 3], strides = []>, i32> {llvm.name = "a"}) attributes {ave.gpu_func = 2 : i32} {
    %c3_i32 = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %c3_i32 : i32 to index
    scf.for %arg1 = %c0 to %0 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      %1 = arith.index_cast %c0_i32 : i32 to index
      %2 = arith.index_cast %arg1 : index to i32
      ave.memref.store %2, %arg0[%1, %arg1] : i32, !ave.memref<!ave.layout<dims = [2, 3], strides = []>, i32>
    }
    return
  }
})";
    TestLLVMCompilation(mlirCode);
}

TEST_F(LLVMTargetTest, GPUBuiltins) {
    const std::string mlirCode = R"(
module {
  func.func @gpu_test(%arg0: !ave.memref<!ave.layout<dims = [32, 32], strides = []>, i32> {llvm.name = "a"}) attributes {ave.gpu_func = 2 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %block_id_x = gpu.block_id  x
    %c0_i32_0 = arith.constant 0 : i32
    %thread_id_x = gpu.thread_id  x
    %c0_i32_1 = arith.constant 0 : i32
    %0 = arith.index_cast %c0_i32_1 : i32 to index
    %1 = arith.index_cast %thread_id_x : index to i32
    ave.memref.store %1, %arg0[%block_id_x, %0] : i32, !ave.memref<!ave.layout<dims = [32, 32], strides = []>, i32>
    return
  }
})";
    TestLLVMCompilation(mlirCode);
}

TEST_F(LLVMTargetTest, MemrefCast) {
    const std::string mlirCode = R"(module {
  func.func @memref_cast_test(%arg0: !ave.memref<!ave.layout<dims = [16], strides = []>, f32> {llvm.name = "a"}) attributes {ave.gpu_func = 2 : i32} {
    %c4_i32 = arith.constant 4 : i32
    %c4_i32_0 = arith.constant 4 : i32
    %0 = ave.make_int_tuple %c4_i32, %c4_i32_0 {is_tuple = true} : i32, i32 -> none
    %c4_i32_1 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = ave.make_int_tuple %c4_i32_1, %c1_i32 {is_tuple = true} : i32, i32 -> none
    %layout1 = ave.make_layout %0, %1 : (none, none) -> !ave.layout
    %2 = ave.memref.cast %arg0, %layout1 : !ave.layout : !ave.memref<!ave.layout<dims = [16], strides = []>, f32> to !ave.memref<!ave.layout<dims = [4, 4], strides = []>, f32>
    %cst = arith.constant 1.000000e+00 : f64
    %3 = arith.truncf %cst : f64 to f32
    %c0_i32 = arith.constant 0 : i32
    %4 = arith.index_cast %c0_i32 : i32 to index
    %c0_i32_2 = arith.constant 0 : i32
    %5 = arith.index_cast %c0_i32_2 : i32 to index
    ave.memref.store %3, %2[%4, %5] : f32, !ave.memref<!ave.layout<dims = [4, 4], strides = []>, f32>
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = ave.make_int_tuple %c2_i32, %c8_i32 {is_tuple = true} : i32, i32 -> none
    %c8_i32_3 = arith.constant 8 : i32
    %c1_i32_4 = arith.constant 1 : i32
    %7 = ave.make_int_tuple %c8_i32_3, %c1_i32_4 {is_tuple = true} : i32, i32 -> none
    %layout2 = ave.make_layout %6, %7 : (none, none) -> !ave.layout
    %8 = ave.memref.cast %arg0, %layout2 : !ave.layout : !ave.memref<!ave.layout<dims = [16], strides = []>, f32> to !ave.memref<!ave.layout<dims = [2, 8], strides = []>, f32>
    %cst_5 = arith.constant 2.000000e+00 : f64
    %9 = arith.truncf %cst_5 : f64 to f32
    %c1_i32_6 = arith.constant 1 : i32
    %10 = arith.index_cast %c1_i32_6 : i32 to index
    %c0_i32_7 = arith.constant 0 : i32
    %11 = arith.index_cast %c0_i32_7 : i32 to index
    ave.memref.store %9, %8[%10, %11] : f32, !ave.memref<!ave.layout<dims = [2, 8], strides = []>, f32>
    return
  }
})";
    TestLLVMCompilation(mlirCode);
}

TEST_F(LLVMTargetTest, VectorLoadStore) {
    const std::string mlirCode = R"(
module {
  func.func @vector_load_store(%arg0: !ave.memref<!ave.layout<dims = [4], strides = []>, i32> {llvm.name = "src"}, %arg1: !ave.memref<!ave.layout<dims = [4], strides = []>, i32> {llvm.name = "dst"}) attributes {ave.gpu_func = 2 : i32} {
    %c0 = arith.constant 0 : index
    %0 = ave.memref.load_vec %arg0[%c0] : !ave.memref<!ave.layout<dims = [4], strides = []>, i32> -> vector<4xi32>
    ave.memref.store %0, %arg1[%c0] : vector<4xi32>, !ave.memref<!ave.layout<dims = [4], strides = []>, i32>
    return
  }
})";

    std::string mlirDump;
    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, &mlirDump, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);

    EXPECT_NE(mlirDump.find("ave.memref.load_vec"), std::string::npos);
    EXPECT_NE(mlirDump.find("vector<4xi32>"), std::string::npos);
    EXPECT_NE(llvmDump.find("load <4 x i32>"), std::string::npos);
    EXPECT_NE(llvmDump.find("store <4 x i32>"), std::string::npos);
}

TEST_F(LLVMTargetTest, ExtractAlignedPointerPreservesSubviewOffset) {
    const std::string mlirCode = R"(
module {
  func.func @extract_ptr_from_subview(%arg0: !ave.memref<!ave.layout<dims = [32], strides = [1]>, bf16> {llvm.name = "base"}, %arg1: !ave.memref<!ave.layout<dims = [1], strides = []>, i64> {llvm.name = "out"}, %arg2: i32 {llvm.name = "row"}) attributes {ave.gpu_func = 2 : i32} {
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.muli %arg2, %c8_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %2 = ave.memref.subview %arg0[%1] [%c8] [%c1] : !ave.memref<!ave.layout<dims = [32], strides = [1]>, bf16> -> !ave.memref<!ave.layout<dims = [8], strides = [1]>, bf16>
    %3 = ave.memref.extract_aligned_pointer_as_index %2 : !ave.memref<!ave.layout<dims = [8], strides = [1]>, bf16> -> index
    %4 = arith.index_cast %3 : index to i64
    %c0 = arith.constant 0 : index
    ave.memref.store %4, %arg1[%c0] : i64, !ave.memref<!ave.layout<dims = [1], strides = []>, i64>
    return
  }
})";

    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);

    EXPECT_NE(llvmDump.find("ptrtoint ptr"), std::string::npos);
    EXPECT_TRUE(llvmDump.find("add i64") != std::string::npos ||
                llvmDump.find("add i32") != std::string::npos ||
                llvmDump.find("getelementptr i8") != std::string::npos)
        << llvmDump;
    EXPECT_NE(llvmDump.find("store i64"), std::string::npos);
}

TEST_F(LLVMTargetTest, SharedMemory) {
    const std::string mlirCode = R"(module {
  func.func @test_shared_memory_basic(%arg0: !ave.memref<!ave.layout<dims = [32], strides = []>, i32> {llvm.name = "input_data"}, %arg1: !ave.memref<!ave.layout<dims = [32], strides = []>, i32> {llvm.name = "output_data"}) attributes {ave.gpu_func = 2 : i32} {
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %alloca = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [32], strides = []>, i32, #gpu.address_space<workgroup>>
    %thread_id_x = gpu.thread_id  x
    %alloca_0 = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>>
    ave.memref.store %thread_id_x, %alloca_0[%c0] : index, !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>>
    %0 = ave.memref.load %alloca_0[%c0] : !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>> -> index
    %1 = ave.memref.load %arg0[%0] : !ave.memref<!ave.layout<dims = [32], strides = []>, i32> -> i32
    %2 = ave.memref.load %alloca_0[%c0] : !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>> -> index
    ave.memref.store %1, %alloca[%2] : i32, !ave.memref<!ave.layout<dims = [32], strides = []>, i32, #gpu.address_space<workgroup>>
    gpu.barrier
    %3 = ave.memref.load %alloca_0[%c0] : !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>> -> index
    %4 = arith.index_cast %3 : index to i32
    %5 = arith.addi %4, %c1_i32 : i32
    %6 = arith.remsi %5, %c32_i32 : i32
    %alloca_1 = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>>
    ave.memref.store %6, %alloca_1[%c0] : i32, !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>>
    %7 = ave.memref.load %alloca_1[%c0] : !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>> -> i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = ave.memref.load %alloca[%8] : !ave.memref<!ave.layout<dims = [32], strides = []>, i32, #gpu.address_space<workgroup>> -> i32
    %10 = ave.memref.load %alloca_0[%c0] : !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>> -> index
    ave.memref.store %9, %arg1[%10] : i32, !ave.memref<!ave.layout<dims = [32], strides = []>, i32>
    return
  }
})";
    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);
    bool hasByteSharedGlobal =
        llvmDump.find("addrspace(3) global [128 x i8] undef") !=
        std::string::npos;
    bool hasTypedSharedGlobal =
        llvmDump.find("addrspace(3) global [32 x i32] undef") !=
        std::string::npos;
    EXPECT_TRUE(hasByteSharedGlobal || hasTypedSharedGlobal);
    EXPECT_NE(llvmDump.find("store i32 %6, ptr addrspace(3) %7, align 4"),
              std::string::npos);
    EXPECT_EQ(llvmDump.find("align 1, addrspace(5)"), std::string::npos);
}

TEST_F(LLVMTargetTest, SharedMemoryUnionStyleSubviews) {
    const std::string mlirCode = R"(module {
  func.func @test_shared_memory_union_subviews(%arg0: !ave.memref<!ave.layout<dims = [8], strides = [1]>, bf16> {llvm.name = "output_data"}) attributes {ave.gpu_func = 2 : i32} {
    %shm = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [32], strides = []>, i8, #gpu.address_space<workgroup>>

    %c8_i32 = arith.constant 8 : i32
    %shape_i32 = ave.make_int_tuple %c8_i32 {is_tuple = true} : i32 -> none
    %c1_i32 = arith.constant 1 : i32
    %stride_i32 = ave.make_int_tuple %c1_i32 {is_tuple = true} : i32 -> none
    %layout_i32 = ave.make_layout %shape_i32, %stride_i32 : (none, none) -> !ave.layout
    %shm_i32 = ave.memref.cast %shm, %layout_i32 : !ave.layout : !ave.memref<!ave.layout<dims = [32], strides = []>, i8, #gpu.address_space<workgroup>> to !ave.memref<!ave.layout<dims = [8], strides = [1]>, i32, #gpu.address_space<workgroup>>

    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %left = ave.memref.subview %shm_i32[%c0] [%c4] [%c1] : !ave.memref<!ave.layout<dims = [8], strides = [1]>, i32, #gpu.address_space<workgroup>> -> !ave.memref<!ave.layout<dims = [4], strides = [1]>, i32, #gpu.address_space<workgroup>>
    %right = ave.memref.subview %shm_i32[%c4] [%c4] [%c1] : !ave.memref<!ave.layout<dims = [8], strides = [1]>, i32, #gpu.address_space<workgroup>> -> !ave.memref<!ave.layout<dims = [4], strides = [1]>, i32, #gpu.address_space<workgroup>>

    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %shape_bf16 = ave.make_int_tuple %c2_i32, %c4_i32 {is_tuple = true} : i32, i32 -> none
    %c4_i32_1 = arith.constant 4 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %stride_bf16 = ave.make_int_tuple %c4_i32_1, %c1_i32_1 {is_tuple = true} : i32, i32 -> none
    %layout_bf16 = ave.make_layout %shape_bf16, %stride_bf16 : (none, none) -> !ave.layout
    %left_bf16 = ave.memref.cast %left, %layout_bf16 : !ave.layout : !ave.memref<!ave.layout<dims = [4], strides = [1]>, i32, #gpu.address_space<workgroup>> to !ave.memref<!ave.layout<dims = [2, 4], strides = [4, 1]>, bf16, #gpu.address_space<workgroup>>
    %right_bf16 = ave.memref.cast %right, %layout_bf16 : !ave.layout : !ave.memref<!ave.layout<dims = [4], strides = [1]>, i32, #gpu.address_space<workgroup>> to !ave.memref<!ave.layout<dims = [2, 4], strides = [4, 1]>, bf16, #gpu.address_space<workgroup>>

    %thread_id_x = gpu.thread_id x
    %tid_slot = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>>
    ave.memref.store %thread_id_x, %tid_slot[%c0] : index, !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>>

    %tid = ave.memref.load %tid_slot[%c0] : !ave.memref<!ave.layout<dims = [1], strides = []>, index, #gpu.address_space<private>> -> index
    %tid_i32 = arith.index_cast %tid : index to i32
    %c0_i32 = arith.constant 0 : i32
    %is_lane0 = arith.cmpi eq, %tid_i32, %c0_i32 : i32
    scf.if %is_lane0 {
      %one_f64 = arith.constant 1.000000e+00 : f64
      %two_f64 = arith.constant 2.000000e+00 : f64
      %three_f64 = arith.constant 3.000000e+00 : f64
      %four_f64 = arith.constant 4.000000e+00 : f64
      %one = arith.truncf %one_f64 : f64 to bf16
      %two = arith.truncf %two_f64 : f64 to bf16
      %three = arith.truncf %three_f64 : f64 to bf16
      %four = arith.truncf %four_f64 : f64 to bf16
      ave.memref.store %one, %left_bf16[%c0, %c0] : bf16, !ave.memref<!ave.layout<dims = [2, 4], strides = [4, 1]>, bf16, #gpu.address_space<workgroup>>
      ave.memref.store %two, %left_bf16[%c0, %c1] : bf16, !ave.memref<!ave.layout<dims = [2, 4], strides = [4, 1]>, bf16, #gpu.address_space<workgroup>>
      ave.memref.store %three, %right_bf16[%c0, %c0] : bf16, !ave.memref<!ave.layout<dims = [2, 4], strides = [4, 1]>, bf16, #gpu.address_space<workgroup>>
      ave.memref.store %four, %right_bf16[%c0, %c1] : bf16, !ave.memref<!ave.layout<dims = [2, 4], strides = [4, 1]>, bf16, #gpu.address_space<workgroup>>
    }

    gpu.barrier

    %c4_i32_2 = arith.constant 4 : i32
    %lt4 = arith.cmpi slt, %tid_i32, %c4_i32_2 : i32
    scf.if %lt4 {
      %value = ave.memref.load %left_bf16[%c0, %tid] : !ave.memref<!ave.layout<dims = [2, 4], strides = [4, 1]>, bf16, #gpu.address_space<workgroup>> -> bf16
      ave.memref.store %value, %arg0[%tid] : bf16, !ave.memref<!ave.layout<dims = [8], strides = [1]>, bf16>
    }

    %ge4 = arith.cmpi sge, %tid_i32, %c4_i32_2 : i32
    %c8_i32_1 = arith.constant 8 : i32
    %lt8 = arith.cmpi slt, %tid_i32, %c8_i32_1 : i32
    %in_right = arith.andi %ge4, %lt8 : i1
    scf.if %in_right {
      %tid_minus4 = arith.subi %tid_i32, %c4_i32_2 : i32
      %tid_right = arith.index_cast %tid_minus4 : i32 to index
      %value = ave.memref.load %right_bf16[%c0, %tid_right] : !ave.memref<!ave.layout<dims = [2, 4], strides = [4, 1]>, bf16, #gpu.address_space<workgroup>> -> bf16
      ave.memref.store %value, %arg0[%tid] : bf16, !ave.memref<!ave.layout<dims = [8], strides = [1]>, bf16>
    }
    return
  }
})";
    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);
    EXPECT_NE(llvmDump.find("addrspace(3) global [32 x i8] undef"),
              std::string::npos);
    bool hasStoreAlign16 =
        llvmDump.find(
            "store <2 x bfloat> <bfloat 0xR3F80, bfloat 0xR4000>, "
            "ptr addrspace(3) @__wg_test_shared_memory_union_subviews_0, "
            "align 16") != std::string::npos;
    bool hasStoreAlign128 =
        llvmDump.find(
            "store <2 x bfloat> <bfloat 0xR3F80, bfloat 0xR4000>, "
            "ptr addrspace(3) @__wg_test_shared_memory_union_subviews_0, "
            "align 128") != std::string::npos;
    EXPECT_TRUE(hasStoreAlign16 || hasStoreAlign128);
}

TEST_F(LLVMTargetTest, SharedMemorySubviewLayoutCastLowers) {
    const std::string mlirCode = R"(module {
  func.func @test_shared_memory_subview_layout_cast(%arg0: !ave.memref<!ave.layout<dims = [2], strides = [1]>, i32> {llvm.name = "output_data"}) attributes {ave.gpu_func = 2 : i32} {
    %shm = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [64], strides = []>, i8, #gpu.address_space<workgroup>>

    %c16_i32 = arith.constant 16 : i32
    %shape_i32 = ave.make_int_tuple %c16_i32 {is_tuple = true} : i32 -> none
    %c1_i32 = arith.constant 1 : i32
    %stride_i32 = ave.make_int_tuple %c1_i32 {is_tuple = true} : i32 -> none
    %layout_i32 = ave.make_layout %shape_i32, %stride_i32 : (none, none) -> !ave.layout
    %shm_i32 = ave.memref.cast %shm, %layout_i32 : !ave.layout : !ave.memref<!ave.layout<dims = [64], strides = []>, i8, #gpu.address_space<workgroup>> to !ave.memref<!ave.layout<dims = [16], strides = [1]>, i32, #gpu.address_space<workgroup>>

    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %sub = ave.memref.subview %shm_i32[%c8] [%c4] [%c1] : !ave.memref<!ave.layout<dims = [16], strides = [1]>, i32, #gpu.address_space<workgroup>> -> !ave.memref<!ave.layout<dims = [4], strides = [1]>, i32, #gpu.address_space<workgroup>>

    %c2_i32 = arith.constant 2 : i32
    %shape_view = ave.make_int_tuple %c2_i32, %c2_i32 {is_tuple = true} : i32, i32 -> none
    %stride_view = ave.make_int_tuple %c2_i32, %c1_i32 {is_tuple = true} : i32, i32 -> none
    %layout_view = ave.make_layout %shape_view, %stride_view : (none, none) -> !ave.layout
    %words = ave.memref.cast %sub, %layout_view : !ave.layout : !ave.memref<!ave.layout<dims = [4], strides = [1]>, i32, #gpu.address_space<workgroup>> to !ave.memref<!ave.layout<dims = [2, 2], strides = [2, 1]>, i32, #gpu.address_space<workgroup>>

    %thread_id_x = gpu.thread_id x
    %tid_i32 = arith.index_cast %thread_id_x : index to i32
    %c0_i32 = arith.constant 0 : i32
    %is_lane0 = arith.cmpi eq, %tid_i32, %c0_i32 : i32
    %c0 = arith.constant 0 : index
    scf.if %is_lane0 {
      %c11_i32 = arith.constant 11 : i32
      ave.memref.store %c11_i32, %words[%c0, %c0] : i32, !ave.memref<!ave.layout<dims = [2, 2], strides = [2, 1]>, i32, #gpu.address_space<workgroup>>
      %value00 = ave.memref.load %words[%c0, %c0] : !ave.memref<!ave.layout<dims = [2, 2], strides = [2, 1]>, i32, #gpu.address_space<workgroup>> -> i32
      ave.memref.store %value00, %arg0[%c0] : i32, !ave.memref<!ave.layout<dims = [2], strides = [1]>, i32>
      %c1_out = arith.constant 1 : index
      %value01 = ave.memref.load %words[%c0, %c1_out] : !ave.memref<!ave.layout<dims = [2, 2], strides = [2, 1]>, i32, #gpu.address_space<workgroup>> -> i32
      ave.memref.store %value01, %arg0[%c1_out] : i32, !ave.memref<!ave.layout<dims = [2], strides = [1]>, i32>
    }
    return
  }
})";
    TestLLVMCompilation(mlirCode);
}

TEST_F(LLVMTargetTest, MemrefCastFoldedScalarLayoutDimsStayStatic) {
    const std::string mlirCode = R"(module {
  func.func @test_folded_scalar_layout_cast(%arg0: !ave.memref<!ave.layout<dims = [1], strides = [1]>, i64> {llvm.name = "out"}) attributes {ave.gpu_func = 2 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c7_i64 = arith.constant 7 : i64

    %slot4 = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>>
    %slot64 = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>>
    ave.memref.store %c4_i32, %slot4[%c0] : i32, !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>>
    ave.memref.store %c64_i32, %slot64[%c0] : i32, !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>>
    %four = ave.memref.load %slot4[%c0] : !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>> -> i32
    %sixtyfour = ave.memref.load %slot64[%c0] : !ave.memref<!ave.layout<dims = [1], strides = []>, i32, #gpu.address_space<private>> -> i32
    %stride0 = arith.muli %four, %c128_i32 : i32
    %stride3 = arith.divui %sixtyfour, %four : i32

    %shape = ave.make_int_tuple %four, %four, %c2_i32, %c32_i32, %c2_i32 {is_tuple = true} : i32, i32, i32, i32, i32 -> none
    %stride = ave.make_int_tuple %stride0, %four, %c2_i32, %stride3, %c1_i32 {is_tuple = true} : i32, i32, i32, i32, i32 -> none
    %layout = ave.make_layout %shape, %stride : (none, none) -> !ave.layout

    %shm = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [16384], strides = []>, i8, #gpu.address_space<workgroup>>
    %words = ave.memref.cast %shm, %layout : !ave.layout : !ave.memref<!ave.layout<dims = [16384], strides = []>, i8, #gpu.address_space<workgroup>> to !ave.memref<!ave.layout<dims = [4, 4, 2, 32, 2], strides = [512, 4, 2, 16, 1]>, i64, #gpu.address_space<workgroup>>

    ave.memref.store %c7_i64, %words[%c0, %c0, %c0, %c0, %c0] : i64, !ave.memref<!ave.layout<dims = [4, 4, 2, 32, 2], strides = [512, 4, 2, 16, 1]>, i64, #gpu.address_space<workgroup>>
    %value = ave.memref.load %words[%c0, %c0, %c0, %c0, %c0] : !ave.memref<!ave.layout<dims = [4, 4, 2, 32, 2], strides = [512, 4, 2, 16, 1]>, i64, #gpu.address_space<workgroup>> -> i64
    ave.memref.store %value, %arg0[%c0] : i64, !ave.memref<!ave.layout<dims = [1], strides = [1]>, i64>
    return
  }
})";

    std::string lowered = LowerAveLangToMemRef(mlirCode);
    ASSERT_FALSE(lowered.empty());
    EXPECT_EQ(lowered.find("memref<4x?x2x32x2xi64"), std::string::npos);
    EXPECT_EQ(lowered.find(
                  "static_sizes = array<i64: 4, 4, 2, 32, 2, 4, 4, 2, 32, 2>"),
              std::string::npos);
    EXPECT_NE(lowered.find("shape_pattern = [4, 4, 2, 32, 2]"),
              std::string::npos);
    EXPECT_NE(lowered.find("stride_pattern = [512, 4, 2, 16, 1]"),
              std::string::npos);

    TestLLVMCompilation(mlirCode);
}

#if defined(WITH_AMDGPU)
TEST_F(LLVMTargetTest, AMDGPUSchedGroupBarrierLowers) {
    const std::string mlirCode = R"(
module {
  gpu.module @kernels {
    gpu.func @sched_group_barrier_kernel() kernel {
      rocdl.sched.group.barrier 2, 4, 1
      gpu.return
    }
  }
}
    )";

    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);

    EXPECT_NE(llvmDump.find("llvm.amdgcn.sched.group.barrier"),
              std::string::npos);
}

TEST_F(LLVMTargetTest, AMDGPUExp2LowersToIntrinsic) {
    const std::string mlirCode = R"(
module {
  func.func @exp2_kernel(%arg0: !ave.memref<!ave.layout<dims = [1], strides = []>, f32> {llvm.name = "out"}, %arg1: f32) attributes {ave.gpu_func = 2 : i32} {
    %c0 = arith.constant 0 : index
    %0 = math.exp2 %arg1 : f32
    ave.memref.store %0, %arg0[%c0] : f32, !ave.memref<!ave.layout<dims = [1], strides = []>, f32>
    return
  }
}
    )";

    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);

    EXPECT_NE(llvmDump.find("llvm.amdgcn.exp2.f32"), std::string::npos);
}

TEST_F(LLVMTargetTest, AMDGPUExpLowersToIntrinsic) {
    const std::string mlirCode = R"(
module {
  func.func @exp_kernel(%arg0: !ave.memref<!ave.layout<dims = [1], strides = []>, f32> {llvm.name = "out"}, %arg1: f32) attributes {ave.gpu_func = 2 : i32} {
    %c0 = arith.constant 0 : index
    %0 = math.exp %arg1 : f32
    ave.memref.store %0, %arg0[%c0] : f32, !ave.memref<!ave.layout<dims = [1], strides = []>, f32>
    return
  }
}
    )";

    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);

    EXPECT_NE(llvmDump.find("llvm.exp.f32"), std::string::npos);
}

TEST_F(LLVMTargetTest, AMDGPUTanhLowersToIntrinsic) {
    const std::string mlirCode = R"(
module {
  func.func @tanh_kernel(%arg0: !ave.memref<!ave.layout<dims = [1], strides = []>, f32> {llvm.name = "out"}, %arg1: f32) attributes {ave.gpu_func = 2 : i32} {
    %c0 = arith.constant 0 : index
    %0 = math.tanh %arg1 : f32
    ave.memref.store %0, %arg0[%c0] : f32, !ave.memref<!ave.layout<dims = [1], strides = []>, f32>
    return
  }
}
    )";

    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);

    EXPECT_NE(llvmDump.find("__ocml_tanh_f32"), std::string::npos);
}

TEST_F(LLVMTargetTest, AMDGPUShuffleIdxLowers) {
    const std::string mlirCode = R"(
module {
  gpu.module @kernels {
    gpu.func @shuffle_kernel(%arg0: memref<1xi32>, %arg1: i32, %arg2: i32) kernel {
      %c64 = arith.constant 64 : i32
      %0, %1 = gpu.shuffle idx %arg2, %arg1, %c64 : i32
      %c0 = arith.constant 0 : index
      memref.store %0, %arg0[%c0] : memref<1xi32>
      gpu.return
    }
  }
}
    )";

    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);

    EXPECT_NE(llvmDump.find("store i32 %2"), std::string::npos) << llvmDump;
}

TEST_F(LLVMTargetTest, AMDGPUShuffleAllModesLower) {
    const std::string mlirCode = R"(
module {
  gpu.module @kernels {
    gpu.func @shuffle_kernel(%arg0: memref<4xi32>, %arg1: i32) kernel {
      %c1 = arith.constant 1 : i32
      %c64 = arith.constant 64 : i32
      %v0, %ok0 = gpu.shuffle idx %arg1, %c1, %c64 : i32
      %v1, %ok1 = gpu.shuffle xor %arg1, %c1, %c64 : i32
      %v2, %ok2 = gpu.shuffle up %arg1, %c1, %c64 : i32
      %v3, %ok3 = gpu.shuffle down %arg1, %c1, %c64 : i32
      %c0 = arith.constant 0 : index
      %c1_idx = arith.constant 1 : index
      %c2_idx = arith.constant 2 : index
      %c3_idx = arith.constant 3 : index
      memref.store %v0, %arg0[%c0] : memref<4xi32>
      memref.store %v1, %arg0[%c1_idx] : memref<4xi32>
      memref.store %v2, %arg0[%c2_idx] : memref<4xi32>
      memref.store %v3, %arg0[%c3_idx] : memref<4xi32>
      gpu.return
    }
  }
}
    )";

    std::string llvmDump;
    auto llvmModule = CompileToLLVM(mlirCode, nullptr, &llvmDump);
    ASSERT_NE(llvmModule, nullptr);

    EXPECT_NE(llvmDump.find("store <2 x i32>"), std::string::npos) << llvmDump;
}
#endif

TEST_F(LLVMTargetTest, MultipleSharedMemory) {
    const std::string mlirCode = R"(module {
  func.func @test_multiple_shm(%arg0: !ave.memref<!ave.layout<dims = [64], strides = []>, i32>) attributes {ave.gpu_func = 2 : i32} {
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    %c21 = arith.constant 21 : i32
    %shm1 = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [64], strides = []>, i32, #gpu.address_space<workgroup>>
    %shm2 = ave.memref.alloca() : !ave.memref<!ave.layout<dims = [32], strides = []>, f32, #gpu.address_space<workgroup>>
    ave.memref.store %c42, %shm1[%c0] : i32, !ave.memref<!ave.layout<dims = [64], strides = []>, i32, #gpu.address_space<workgroup>>
    %0 = ave.memref.load %shm1[%c0] : !ave.memref<!ave.layout<dims = [64], strides = []>, i32, #gpu.address_space<workgroup>> -> i32
    ave.memref.store %0, %arg0[%c0] : i32, !ave.memref<!ave.layout<dims = [64], strides = []>, i32>
    return
  }
})";
    TestLLVMCompilation(mlirCode);
}

} // namespace causalflow::avelang::target::gpu
