#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Frontend/avelang_parser.h"
#include "IR/ir_context.h"
#include "IR/mlir_generator.h"
#include "IR/symbol_table.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#pragma clang diagnostic pop

#include <gtest/gtest.h>
#include <llvm/Support/MemoryBuffer.h>
#include <set>
#include <vector>

namespace causalflow::avelang::frontend {
namespace cf = causalflow::avelang::dialect;

class DiagnosticHandler : public clang::DiagnosticConsumer {
  public:
    DiagnosticHandler();
    void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                          const clang::Diagnostic &Info) override;
    void RegisterMLIRHandler(mlir::MLIRContext *context);
    std::string
    GetErrorMessages(const clang::SourceManager *SM = nullptr) const;
    std::string GetMLIRErrorMessages() const;
    bool HasMLIRErrors() const;
    void Clear();

  private:
    llvm::SmallVector<std::string, 4> clang_errors_;
    llvm::SmallVector<std::string, 4> mlir_errors_;
    mlir::DiagnosticEngine::HandlerID mlir_handler_id_;
    bool mlir_handler_registered_ = false;
};

DiagnosticHandler::DiagnosticHandler() = default;

void DiagnosticHandler::HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                                         const clang::Diagnostic &Info) {
    if (level >= clang::DiagnosticsEngine::Error) {
        std::string message = "Error";
        if (Info.getLocation().isValid() && Info.hasSourceManager()) {
            auto &SM = Info.getSourceManager();
            clang::FullSourceLoc loc(SM.getFileLoc(Info.getLocation()), SM);
            if (loc.isValid()) {
                unsigned line = loc.getSpellingLineNumber();
                unsigned column = loc.getSpellingColumnNumber();
                if (line != 0 && column != 0) {
                    message += " at line " + std::to_string(line) +
                               ", column " + std::to_string(column);
                }
            }
        }
        llvm::SmallString<128> formatted;
        Info.FormatDiagnostic(formatted);
        message += ": " + formatted.str().str();
        clang_errors_.push_back(std::move(message));
    }
}

void DiagnosticHandler::RegisterMLIRHandler(mlir::MLIRContext *context) {
    if (!context) {
        return;
    }

    mlir_errors_.clear();
    mlir_handler_registered_ = true;
    mlir_handler_id_ = context->getDiagEngine().registerHandler(
        [this](mlir::Diagnostic &diag) {
            if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
                std::string msg = diag.str();
                std::string location_str;

                llvm::raw_string_ostream loc_os(location_str);
                diag.getLocation().print(loc_os);

                mlir_errors_.push_back(
                    "MLIR Error: " + msg +
                    (location_str.empty() ? "" : " at " + location_str));
            }
            return mlir::success();
        });
}

std::string
DiagnosticHandler::GetErrorMessages(const clang::SourceManager *SM) const {
    std::string messages;
    for (const auto &error : clang_errors_) {
        messages += error + "\n";
    }
    return messages;
}

std::string DiagnosticHandler::GetMLIRErrorMessages() const {
    std::string result;
    for (const auto &error : mlir_errors_) {
        result += error + "\n";
    }
    return result;
}

bool DiagnosticHandler::HasMLIRErrors() const { return !mlir_errors_.empty(); }

void DiagnosticHandler::Clear() {
    clang_errors_.clear();
    mlir_errors_.clear();
}

class MLIRGeneratorTest : public ::testing::Test {
  protected:
    virtual void SetUp() override;
    void TryParse(const std::string &source_code, ast::ASTNode **root);
    void RunMLIRGenerationTest(const std::string &source_code);
    void
    RunMLIRGenerationTestWithJitDeps(const std::string &source_code,
                                     const std::vector<std::string> &jit_deps);
    void RunMLIRGenerationErrorTest(const std::string &source_code,
                                    const std::string &expected_error);
    void RunMLIRGenerationErrorTestWithJitDeps(
        const std::string &source_code, const std::string &expected_error,
        const std::vector<std::string> &jit_deps);
    void RegisterJitDependencies(ast::ASTNode *root,
                                 ir::MLIRGenerator &generator,
                                 const std::vector<std::string> &jit_deps);

    llvm::IntrusiveRefCntPtr<ast::ASTContext> context_;
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs_;
    std::unique_ptr<clang::DiagnosticOptions> diagOpts_;
    DiagnosticHandler diagHandler_;
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostics_;
};

void MLIRGeneratorTest::SetUp() {
    context_ = new ast::ASTContext();
    diagIDs_ = new clang::DiagnosticIDs();
    diagOpts_ = std::make_unique<clang::DiagnosticOptions>();
    diagHandler_.Clear();
    diagnostics_ = new basic::DiagnosticManager(&diagHandler_);
}

void MLIRGeneratorTest::TryParse(const std::string &source_code,
                                 ast::ASTNode **root) {
    AveLangParser parser(context_, diagnostics_);
    auto buffer = llvm::MemoryBuffer::getMemBufferCopy(source_code, "stdin");
    parser.ParseFromBuffer(*buffer, "stdin");

    auto *module = parser.GetModule();
    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_NE(module, nullptr) << diagHandler_.GetErrorMessages(&SM);
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);

    *root = module;
}

void MLIRGeneratorTest::RunMLIRGenerationTest(const std::string &source_code) {
    ast::ASTNode *root;
    TryParse(source_code, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

void MLIRGeneratorTest::RegisterJitDependencies(
    ast::ASTNode *root, ir::MLIRGenerator &generator,
    const std::vector<std::string> &jit_deps) {
    auto *module = llvm::dyn_cast<ast::Module>(root);
    ASSERT_NE(module, nullptr);

    for (const auto &name : jit_deps) {
        ast::FunctionDef *func = nullptr;
        for (auto *stmt : module->GetBody()) {
            if (auto *candidate = llvm::dyn_cast<ast::FunctionDef>(stmt)) {
                if (candidate->GetName() == name) {
                    func = candidate;
                    break;
                }
            }
        }
        ASSERT_NE(func, nullptr) << "JIT dependency not found: " << name;
        generator.RegisterJitDependency(func);
    }
}

void MLIRGeneratorTest::RunMLIRGenerationTestWithJitDeps(
    const std::string &source_code, const std::vector<std::string> &jit_deps) {
    ast::ASTNode *root;
    TryParse(source_code, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    RegisterJitDependencies(root, generator, jit_deps);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

void MLIRGeneratorTest::RunMLIRGenerationErrorTest(
    const std::string &source_code, const std::string &expected_error) {
    ast::ASTNode *root;
    TryParse(source_code, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    // Clear any previous diagnostic messages
    diagHandler_.Clear();

    // Register MLIR diagnostic handler to capture verifier errors
    diagHandler_.RegisterMLIRHandler(ir_context->GetMLIRContext());

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir_module = generator.Generate(root);

    // Verify the MLIR module to trigger dialect verifiers
    // This will catch errors from emitOpError in dialect verifiers
    mlir::LogicalResult result = mlir::verify(mlir_module);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();

    // Combine error messages from both clang and MLIR diagnostics
    std::string error_messages = diagHandler_.GetErrorMessages(&SM);
    if (diagHandler_.HasMLIRErrors()) {
        error_messages += diagHandler_.GetMLIRErrorMessages();
    }

    // We expect an error to occur (either from clang or MLIR verifier)
    bool has_error = diagnostics_->GetEngine()->hasErrorOccurred() ||
                     diagHandler_.HasMLIRErrors() || mlir::failed(result);
    ASSERT_TRUE(has_error) << "Expected an error but none occurred";

    // Check that the error message contains the expected text
    ASSERT_FALSE(error_messages.empty()) << "No error messages captured";
    ASSERT_NE(error_messages.find(expected_error), std::string::npos)
        << "Expected error message '" << expected_error
        << "' not found in: " << error_messages;
}

void MLIRGeneratorTest::RunMLIRGenerationErrorTestWithJitDeps(
    const std::string &source_code, const std::string &expected_error,
    const std::vector<std::string> &jit_deps) {
    ast::ASTNode *root;
    TryParse(source_code, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    diagHandler_.Clear();
    diagHandler_.RegisterMLIRHandler(ir_context->GetMLIRContext());

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    RegisterJitDependencies(root, generator, jit_deps);
    auto mlir_module = generator.Generate(root);

    mlir::LogicalResult result = mlir::verify(mlir_module);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();

    std::string error_messages = diagHandler_.GetErrorMessages(&SM);
    if (diagHandler_.HasMLIRErrors()) {
        error_messages += diagHandler_.GetMLIRErrorMessages();
    }

    bool has_error = diagnostics_->GetEngine()->hasErrorOccurred() ||
                     diagHandler_.HasMLIRErrors() || mlir::failed(result);
    ASSERT_TRUE(has_error) << "Expected an error but none occurred";

    ASSERT_FALSE(error_messages.empty()) << "No error messages captured";
    ASSERT_NE(error_messages.find(expected_error), std::string::npos)
        << "Expected error message '" << expected_error
        << "' not found in: " << error_messages;
}

TEST_F(MLIRGeneratorTest, GenerateMLIR) {
    static const std::string kSourceCode = R""""(
import avelang
import avelang.language as S

@avelang.jit
def matmul(a: S.i32):
    return
)"""";
    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateKernelFunctionWithReturn) {
    static const std::string kSourceCode = R""""(
import avelang
import avelang.language as S

@avelang.jit
def matmul(a: S.i32) -> S.i32:
    return 24
)"""";
    RunMLIRGenerationErrorTest(
        kSourceCode, "GPU kernel functions cannot explicitly return values.");
}

TEST_F(MLIRGeneratorTest, JitFunctionReturnWithoutTypeAnnotation) {
    static const std::string kSourceCode = R""""(
import avelang
import avelang.language as S

@avelang.jit
def helper(a: S.f32):
    return a * 2.0
)"""";
    RunMLIRGenerationErrorTestWithJitDeps(
        kSourceCode, "has a return value but no return type annotation",
        {"helper"});
}

TEST_F(MLIRGeneratorTest, JitFunctionReturnWithTypeAnnotation) {
    static const std::string kSourceCode = R""""(
import avelang
import avelang.language as S

@avelang.jit
def helper(a: S.f32) -> S.f32:
    return S.convert(a * 2.0, S.f32)

@avelang.jit
def kernel(input_data: S.Tensor((4,), S.f32), output_data: S.Tensor((4,), S.f32)):
    for i in S.range(4):
        output_data[i] = helper(input_data[i])
)"""";
    RunMLIRGenerationTestWithJitDeps(kSourceCode, {"helper"});
}

TEST_F(MLIRGeneratorTest, JitSymbolTableScopesForLazyAndNestedFunctions) {
    static const std::string kKernelSource = R""""(
import avelang
import avelang.language as S

@avelang.jit
def kernel(p: S.Tensor((1,), S.i32)):
    def inner(y: S.Tensor((1,), S.i32)):
        y[0] = 2
    inner(p)
    helper(p)
)"""";

    static const std::string kHelperSource = R""""(
def helper(x: S.Tensor((1,), S.i32)):
    x[0] = 1
)"""";

    ast::ASTNode *helper_root;
    TryParse(kHelperSource, &helper_root);
    ASSERT_NE(helper_root, nullptr);

    auto *helper_module = llvm::dyn_cast<ast::Module>(helper_root);
    ASSERT_NE(helper_module, nullptr);

    ast::FunctionDef *helper_func = nullptr;
    for (auto *stmt : helper_module->GetBody()) {
        if (auto *candidate = llvm::dyn_cast<ast::FunctionDef>(stmt)) {
            helper_func = candidate;
            break;
        }
    }
    ASSERT_NE(helper_func, nullptr);

    ast::ASTNode *kernel_root;
    TryParse(kKernelSource, &kernel_root);
    ASSERT_NE(kernel_root, nullptr);

    auto ir_context = ir::IRContext::Create();
    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    generator.RegisterJitDependency(helper_func);
    auto mlir = generator.Generate(kernel_root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRWithAssignScalarToScalarMemref) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def matmul(a: S.Tensor((2, 3), S.i32)):
    a[0, 0] = 1
)""""";
    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRWithAssignTooMuchIndices) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def matmul(a: S.Tensor((2, 3), S.i32)):
    a[0, 0, 0] = 1
)""""";
    RunMLIRGenerationErrorTest(
        kSourceCode, "Index dimension mismatch in assignment's target");
}

TEST_F(MLIRGeneratorTest, GenerateMLIRWitjAssignTooFewIndices) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def matmul(a: S.Tensor((2, 3), S.i32)):
    a[0] = 1
)""""";
    RunMLIRGenerationErrorTest(
        kSourceCode, "Index dimension mismatch in assignment's target");
}

TEST_F(MLIRGeneratorTest, GenerateMLIRWithAssignVectorToVectorMemref) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def matmul(a: S.Tensor((64, 4), S.u32), b: S.Tensor((8, 4), S.u32)):
    a[0] = b[0]
)""""";
    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRWithAssignVectorToScalarMemref) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def matmul(a: S.Tensor((64,), S.u32), b: S.Tensor((8, 4), S.u32)):
    a[0] = b[0]
)""""";
    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRBinaryOps) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def binops(a: S.Tensor((2, 3), S.i32), b: S.Tensor((2, 3), S.i32)):
    a[0, 0] = (b[0, 1] + 5) * 2 + 4 - 3 * 4 / 2 % 3
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRUnaryMinusFloatExpr) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def unary_minus_float(out: S.Tensor((1,), S.f32), inp: S.Tensor((1,), S.f32)):
    out[0] = S.exp2(-inp[0])
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundNeg = false;
    bool foundExp2 = false;
    mlir->walk([&](mlir::arith::NegFOp) { foundNeg = true; });
    mlir->walk([&](mlir::math::Exp2Op) { foundExp2 = true; });

    EXPECT_TRUE(foundNeg);
    EXPECT_TRUE(foundExp2);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRTanhExpr) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def tanh_expr(out: S.Tensor((1,), S.f32), inp: S.Tensor((1,), S.f32)):
    out[0] = S.tanh(inp[0])
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundTanh = false;
    mlir->walk([&](mlir::math::TanhOp) { foundTanh = true; });

    EXPECT_TRUE(foundTanh);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRTypePromotion) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def int_types_test(a: S.Tensor((2, 3), S.i64), b: S.Tensor((2, 3), S.i64)):
    a[0, 0] = b[0, 1] + 42
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRBitwiseOps) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def bitwise_and_test(a: S.Tensor((2, 3), S.i32), b: S.Tensor((2, 3), S.i32)):
    a[0, 0] = b[0, 1] & (15 | 8 ^ 255 << 2) >> 1
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRCast) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def cast_test(a: S.Tensor((2, 3), S.i32), b: S.Tensor((2, 3), S.i32)):
    a[0, 0] = b[0, 1] + S.convert(15, S.i32)

)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRBitcast) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def bitcast_test(a: S.Tensor((2, 3), S.f32), b: S.Tensor((2, 3), S.f32)):
    a[0, 0] = b[0, 1] + S.bitcast(0x3f800000, S.f32)

)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRF32Tensor) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def f32_test(a: S.Tensor((2, 3), S.f32)):
    # Python float literal is f64, so we need to convert to f32
    a[0, 0] = S.convert(1.0, S.f32)

)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRIf) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def if_test(a: S.Tensor((2, 3), S.i32)):
    if a[0, 0] > 0:
        a[0, 0] = 1
    else:
        a[0, 0] = 0

)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRIfStmt) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def if_test(a: S.Tensor((32,), S.i32)):
    c = 5
    if S.thread_id(0) > 5:
        c = 1
    a[S.thread_id(0)] = c
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRJitFunctionCall) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def helper(a: S.Tensor((32,), S.i32)):
    a[0] = 42

@avelang.jit
def kernel(a: S.Tensor((32,), S.i32)):
    helper(a)
)""""";

    RunMLIRGenerationTestWithJitDeps(kSourceCode, {"helper"});
}

TEST_F(MLIRGeneratorTest, SpecializeJitFunctionByAddressSpace) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def helper(a: S.Tensor((4,), S.i32)):
    a[0] = 1

@avelang.jit
def kernel(a: S.Tensor((4,), S.i32)):
    shared = S.make_shared((4,), S.i32)
    local = S.make_local((4,), S.i32)
    shared_view = S.view(shared, S.Tensor((4,), S.i32))
    local_view = S.view(local, S.Tensor((4,), S.i32))
    helper(shared_view)
    helper(local_view)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();
    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    RegisterJitDependencies(root, generator, {"helper"});

    auto module = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(module, nullptr);

    mlir::PassManager pm(module.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(module))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)))
        << "MLIR verification failed!";

    mlir::func::FuncOp workgroup_func;
    mlir::func::FuncOp private_func;
    module.walk([&](mlir::func::FuncOp func) {
        auto name = func.getName().str();
        if (name == "kernel_helper__as_workgroup") {
            workgroup_func = func;
        } else if (name == "kernel_helper__as_private") {
            private_func = func;
        }
    });

    ASSERT_TRUE(workgroup_func);
    ASSERT_TRUE(private_func);

    auto check_address_space = [](mlir::func::FuncOp func,
                                  mlir::gpu::AddressSpace expected) {
        ASSERT_EQ(func.getNumArguments(), 1u);
        auto memref_type =
            mlir::dyn_cast<cf::MemRefType>(func.getArgument(0).getType());
        ASSERT_TRUE(memref_type);
        auto memory_space = memref_type.getMemorySpace();
        auto addr_space =
            mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(memory_space);
        ASSERT_TRUE(addr_space);
        EXPECT_EQ(addr_space.getValue(), expected);
    };

    check_address_space(workgroup_func, mlir::gpu::AddressSpace::Workgroup);
    check_address_space(private_func, mlir::gpu::AddressSpace::Private);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRJitFunctionWithReturnCall) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def helper() -> S.i32:
    return 42

@avelang.jit
def kernel(a: S.Tensor((32,), S.i32)):
    a[0] = helper()
)""""";

    RunMLIRGenerationTestWithJitDeps(kSourceCode, {"helper"});
}

TEST_F(MLIRGeneratorTest, GenerateMLIRJitFunctionWithTensorReturn) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def make_pair(a: S.u32, b: S.u32) -> S.Tensor((2,), S.u32):
    pair = S.make_local((2,), S.u32)
    pair[0] = a
    pair[1] = b
    return pair

@avelang.jit
def kernel(out: S.Tensor((2,), S.u32)):
    pair = make_pair(S.convert(3, S.u32), S.convert(5, S.u32))
    out[0] = pair[0]
    out[1] = pair[1]
)""""";

    RunMLIRGenerationTestWithJitDeps(kSourceCode, {"make_pair"});
}

TEST_F(MLIRGeneratorTest, GenerateMLIRJitFunctionWithMultipleTensorReturns) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def make_pairs(a: S.u32, b: S.u32) -> (S.Tensor((2,), S.u32), S.Tensor((2,), S.u32)):
    first = S.make_local((2,), S.u32)
    second = S.make_local((2,), S.u32)
    first[0] = a
    first[1] = b
    second[0] = b
    second[1] = a
    return first, second

@avelang.jit
def kernel(out: S.Tensor((4,), S.u32)):
    first, second = make_pairs(S.convert(3, S.u32), S.convert(5, S.u32))
    out[0] = first[0]
    out[1] = first[1]
    out[2] = second[0]
    out[3] = second[1]
)""""";

    RunMLIRGenerationTestWithJitDeps(kSourceCode, {"make_pairs"});
}

TEST_F(MLIRGeneratorTest, GenerateMLIRNestedFunctionCall) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def kernel(a: S.Tensor((4,), S.f32),
           b: S.Tensor((4,), S.f32),
           c: S.Tensor((4,), S.f32)):
    def helper(x: S.f32, y: S.f32) -> S.f32:
        return x + y
    c[0] = helper(a[0], b[0])
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, MangleNestedPrivateFunctionNames) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def kernel(a: S.Tensor((4,), S.f32),
           b: S.Tensor((4,), S.f32),
           c: S.Tensor((4,), S.f32)):
    def foo(x: S.f32, y: S.f32) -> S.f32:
        def helper(u: S.f32, v: S.f32) -> S.f32:
            return u + v
        return helper(x, y)
    def bar(x: S.f32, y: S.f32) -> S.f32:
        def helper(u: S.f32, v: S.f32) -> S.f32:
            return u * v
        return helper(x, y)
    c[0] = foo(a[0], b[0]) + bar(a[1], b[1])
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();
    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";

    std::set<std::string> func_names;
    mlir.walk([&](mlir::func::FuncOp func) {
        func_names.insert(func.getName().str());
    });

    EXPECT_TRUE(func_names.count("kernel"));
    EXPECT_TRUE(func_names.count("kernel_foo"));
    EXPECT_TRUE(func_names.count("kernel_bar"));
    EXPECT_TRUE(func_names.count("kernel_foo_helper"));
    EXPECT_TRUE(func_names.count("kernel_bar_helper"));
    EXPECT_FALSE(func_names.count("helper"));
}

TEST_F(MLIRGeneratorTest, GenerateMLIRIfExpr) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def if_expr_test(a: S.Tensor((2, 3), S.i32)):
    a[0, 0] = 1 if 5 > 3 else 0
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRIfExprTypeError) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def if_expr_type_error_test(a: S.Tensor((2, 3), S.i32), b: S.Tensor((4, 5), S.f32)):
    a[0, 0] = a if 5 > 3 else b
)""""";

    RunMLIRGenerationErrorTest(
        kSourceCode, "Conditional expression branches have incompatible types");
}

TEST_F(MLIRGeneratorTest, GenerateMLIRForLoop) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def for_loop_test(a: S.Tensor((2, 3), S.i32)):
    for i in S.range(3):
        a[0, i] = i
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRForLoopWithRange2Args) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def for_loop_range2_test(a: S.Tensor((10, 10), S.i32)):
    for i in S.range(1, 5):
        a[0, i] = i
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRForLoopWithRange3Args) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def for_loop_range3_test(a: S.Tensor((10, 10), S.i32)):
    for i in S.range(0, 10, 2):
        a[0, i] = i
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRForLoopWithNestedMinRangeBound) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def for_loop_nested_min_range_test(a: S.Tensor((8, 8), S.i32)):
    for i in S.range(S.min(S.block_id(0) + 1, 4)):
        a[0, i] = i
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRWhileLoop) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def while_loop_test(a: S.Tensor((2, 3), S.i32)):
    i = 0
    while i < 3:
        i += 1
        a[0, i] = 3
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRGPUBuiltins) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def gpu_test(a: S.Tensor((32, 32), S.i32)):
    idx = S.block_id(0)
    a[idx, 0] = S.thread_id(0)
    S.syncthreads()
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRUnsignedMinMaxBuiltins) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def unsigned_min_max(lhs: S.u32, rhs: S.u32):
    lo = S.min(lhs, rhs)
    hi = S.max(lhs, rhs)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundMinUI = false;
    bool foundMaxUI = false;
    mlir->walk([&](mlir::arith::MinUIOp op) {
        foundMinUI = true;
        EXPECT_TRUE(op.getType().isInteger(32));
    });
    mlir->walk([&](mlir::arith::MaxUIOp op) {
        foundMaxUI = true;
        EXPECT_TRUE(op.getType().isInteger(32));
    });

    EXPECT_TRUE(foundMinUI);
    EXPECT_TRUE(foundMaxUI);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRIndexMinMaxBuiltins) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def index_min_max(a: S.Tensor((1,), S.i32)):
    lhs = S.block_id(0)
    rhs = S.block_dim(0)
    lo = S.min(lhs, rhs)
    hi = S.max(lhs, rhs)
    a[0] = S.convert(lo + hi, S.i32)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRUnsignedBitcastRShift) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def unsigned_bitcast_rshift(out: S.Tensor((1,), S.u32), inp: S.Tensor((1,), S.f32)):
    out[0] = S.bitcast(inp[0], S.u32) >> S.convert(16, S.u32)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundUnsignedAttr = false;
    bool foundUnsignedShift = false;
    bool foundSignedShift = false;

    mlir->walk([&](mlir::Operation *op) {
        if (auto unsignedAttr = op->getAttrOfType<mlir::IntegerAttr>(
                "ave.type_info.is_unsigned_integer")) {
            if (unsignedAttr.getInt() == 2) {
                foundUnsignedAttr = true;
            }
        }
    });
    mlir->walk([&](mlir::arith::ShRUIOp) { foundUnsignedShift = true; });
    mlir->walk([&](mlir::arith::ShRSIOp) { foundSignedShift = true; });

    EXPECT_TRUE(foundUnsignedAttr);
    EXPECT_TRUE(foundUnsignedShift);
    EXPECT_FALSE(foundSignedShift);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRUnsignedBitcastConstantFoldShape) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def unsigned_bitcast_shape_fold(
    out: S.Tensor((S.bitcast(-2147483648, S.u32) >> S.convert(31, S.u32),), S.i32)
):
    out[0] = 0
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    mlir::func::FuncOp func;
    mlir.walk([&func](mlir::func::FuncOp f) {
        if (f.getName() == "unsigned_bitcast_shape_fold") {
            func = f;
        }
    });
    ASSERT_TRUE(func);
    ASSERT_EQ(func.getNumArguments(), 1u);

    auto argType =
        mlir::dyn_cast<cf::MemRefType>(func.getArgument(0).getType());
    ASSERT_TRUE(argType);
    ASSERT_EQ(argType.getRank(), 1);
    EXPECT_EQ(argType.getShape()[0], 1);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAllocLocal) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def alloc_local_test():
    local_mem = S.make_local((16, 16), S.f32)
    local_mem[0, 0] = S.convert(1.0, S.f32)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAllocShared) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def alloc_shared_test(input: S.Tensor((32,), S.i32),
                      output: S.Tensor((32,), S.i32)):
    shared_mem = S.make_shared((32,), S.i32)
    idx = S.thread_id(0)
    shared_mem[idx] = input[idx]
    S.syncthreads()
    output[idx] = shared_mem[(idx + 1) % 32]
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAllocSharedWithFoldedShapeExprs) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def alloc_shared_folded_shape_test(input: S.Tensor((64,), S.i32),
                                   output: S.Tensor((64,), S.i32)):
    TILE_M = 32
    TILE_K = 16
    shared_mem = S.make_shared((TILE_M * (TILE_K >> 3), TILE_K >> 2), S.i32)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAxpy) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def axpy(a: S.i32, b: S.i32, x: S.Tensor((64,), S.i32), y: S.Tensor((64,), S.i32)):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    y[idx] = a * x[idx] + b
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRNVVMMMASync) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def mma_test(a: S.Tensor((4,), S.i32),
            b: S.Tensor((2,), S.i32),
            c: S.Tensor((4,), S.f16)):
    c = S.nvvm.mma_16x8x16_f16_f16(a, b, c)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRNVVMLdMatrix) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def ldmatrix_comprehensive_test():
    shared_mem = S.make_shared((8, 8), S.f16)
    result = S.nvvm.ldmatrix_m8n8_x1_b16(shared_mem)    # 1xi32 = 2xf16
    result_4 = S.make_local((4,), S.i32)
    result_4 = S.nvvm.ldmatrix_m8n8_x4_b16(shared_mem)  # 4xi32 = 8xf16
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRNVVMStMatrix) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def stmatrix_comprehensive_test(data1: S.i32,
                               data2: S.Tensor((2,), S.i32),
                               data4: S.Tensor((4,), S.i32)):
    # Test key stmatrix variants
    shared_mem = S.make_shared((8, 8), S.f16)

    # Test x1, x2, x4 variants using function parameters
    S.nvvm.stmatrix_m8n8_x1_b16(shared_mem, data1)
    S.nvvm.stmatrix_m8n8_x2_b16(shared_mem, data2)
    S.nvvm.stmatrix_m8n8_x4_b16(shared_mem, data4)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPUMFMASync) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def mfma_test(a: S.Tensor((4,), S.f16),
             b: S.Tensor((4,), S.f16),
             c: S.Tensor((4,), S.f32)):
    c = S.amdgpu.mfma_16x16x16_f16_f32(a, b, c)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPUMFMASyncWithVectorElement) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def mfma_test(a: S.Tensor((4,), S.f16),
             b: S.Tensor((4,), S.f16),
             c: S.Tensor((4,), S.f32)):
    a_cast = S.view(a, S.Tensor((1, 4, 1), S.f16))
    b_cast = S.view(b, S.Tensor((1, 4, 1), S.f16))
    c = S.amdgpu.mfma_16x16x16_f16_f32(a_cast[0], b_cast[0], c)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPUMFMABF16) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def mfma_bf16_test(a: S.Tensor((4,), S.bf16),
                  b: S.Tensor((4,), S.bf16),
                  c: S.Tensor((4,), S.f32)):
    c = S.amdgpu.mfma_16x16x16_bf16_f32(a, b, c)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPUMFMABF16Alias) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def mfma_bf16_test(a: S.Tensor((4,), S.bf16),
                  b: S.Tensor((4,), S.bf16),
                  c: S.Tensor((4,), S.f32)):
    c = S.amdgpu.mfma_f32_16x16x16_bf16(a, b, c)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPURawBufferLoad) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def raw_buffer_load_test(rsrc: S.Tensor((4,), S.u32),
                        vindex: S.i32,
                        soffset: S.i32,
                        aux: S.i32):
    result1 = S.amdgpu.raw_buffer_load_x1(rsrc, vindex, soffset, aux)
    result2 = S.amdgpu.raw_buffer_load_x2(rsrc, vindex, soffset, aux)
    result4 = S.amdgpu.raw_buffer_load_x4(rsrc, vindex, soffset, aux)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPURawBufferStore) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def raw_buffer_store_test(vdata1: S.i32,
                         vdata2: S.Tensor((2,), S.i32),
                         vdata4: S.Tensor((4,), S.i32),
                         rsrc: S.Tensor((4,), S.u32),
                         vindex: S.i32,
                         soffset: S.i32,
                         aux: S.i32):
    S.amdgpu.raw_buffer_store_x1(vdata1, rsrc, vindex, soffset, aux)
    S.amdgpu.raw_buffer_store_x2(vdata2, rsrc, vindex, soffset, aux)
    S.amdgpu.raw_buffer_store_x4(vdata4, rsrc, vindex, soffset, aux)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPUSchedGroupBarrier) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def sched_group_barrier_test():
    S.amdgpu.sched_group_barrier(2, 4, 1)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool found_sched_group_barrier = false;
    mlir->walk([&](mlir::ROCDL::SchedGroupBarrier op) {
        found_sched_group_barrier = true;
        EXPECT_EQ(op.getMask(), 2u);
        EXPECT_EQ(op.getSize(), 4u);
        EXPECT_EQ(op.getGroupId(), 1u);
    });

    EXPECT_TRUE(found_sched_group_barrier);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPUPerm) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def perm_test(out: S.Tensor((1,), S.u32),
              lhs: S.u32,
              rhs: S.u32,
              sel: S.u32):
    out[0] = S.amdgpu.perm(lhs, rhs, sel)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool found_perm_call = false;
    mlir->walk([&](mlir::func::CallOp op) {
        if (op.getCallee() == "_avelang_amdgpu_llvm_amdgcn_perm") {
            found_perm_call = true;
            EXPECT_EQ(op.getNumOperands(), 3u);
            EXPECT_EQ(op.getNumResults(), 1u);
            EXPECT_TRUE(op.getResult(0).getType().isInteger(32));
        }
    });

    EXPECT_TRUE(found_perm_call);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPURawBufferLoadLds) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def raw_buffer_load_lds_test(rsrc: S.Tensor((4,), S.u32)):
    shared_mem = S.make_shared((16,), S.u32)
    S.amdgpu.raw_buffer_load_x1_lds(rsrc, shared_mem, 4, 0, 0, 16, 0)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundLoadLdsCall = false;
    mlir->walk([&](mlir::func::CallOp op) {
        if (op.getCallee() ==
            "_avelang_amdgpu_llvm_amdgcn_raw_buffer_load_lds_u32") {
            foundLoadLdsCall = true;
            EXPECT_EQ(op.getNumOperands(), 4u);
            EXPECT_EQ(op.getNumResults(), 0u);
        }
    });

    EXPECT_TRUE(foundLoadLdsCall);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRAMDGPUSWaitcnt) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def s_waitcnt_test():
    S.amdgpu.s_waitcnt(0, 7, 15)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundWaitcnt = false;
    mlir->walk([&](mlir::ROCDL::SWaitcntOp op) {
        foundWaitcnt = true;
        EXPECT_EQ(op.getBitfield(), 3952);
    });

    EXPECT_TRUE(foundWaitcnt);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRShuffleIdxF32) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def shuffle_test(index: S.u32, value: S.f32):
    result = S.shuffle(value, index, 32)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundShuffle = false;
    mlir->walk([&](mlir::gpu::ShuffleOp op) {
        foundShuffle = true;
        EXPECT_EQ(op.getMode(), mlir::gpu::ShuffleMode::IDX);
        EXPECT_TRUE(op.getShuffleResult().getType().isF32());
    });

    EXPECT_TRUE(foundShuffle);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRShuffleUpF32) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def shuffle_up_test(offset: S.u32, value: S.f32):
    result = S.shuffle_up(value, offset, 32)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundShuffle = false;
    mlir->walk([&](mlir::gpu::ShuffleOp op) {
        foundShuffle = true;
        EXPECT_EQ(op.getMode(), mlir::gpu::ShuffleMode::UP);
        EXPECT_TRUE(op.getShuffleResult().getType().isF32());
    });

    EXPECT_TRUE(foundShuffle);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRShuffleDownF32) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def shuffle_down_test(offset: S.u32, value: S.f32):
    result = S.shuffle_down(value, offset, 32)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundShuffle = false;
    mlir->walk([&](mlir::gpu::ShuffleOp op) {
        foundShuffle = true;
        EXPECT_EQ(op.getMode(), mlir::gpu::ShuffleMode::DOWN);
        EXPECT_TRUE(op.getShuffleResult().getType().isF32());
    });

    EXPECT_TRUE(foundShuffle);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRShuffleXorF32) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def shuffle_xor_test(mask: S.u32, value: S.f32):
    result = S.shuffle_xor(value, mask, 32)
)""""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    auto mlir = generator.Generate(root);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundShuffle = false;
    mlir->walk([&](mlir::gpu::ShuffleOp op) {
        foundShuffle = true;
        EXPECT_EQ(op.getMode(), mlir::gpu::ShuffleMode::XOR);
        EXPECT_TRUE(op.getShuffleResult().getType().isF32());
    });

    EXPECT_TRUE(foundShuffle);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRVectorSubscriptRead) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def vector_subscript_read_test(rsrc: S.Tensor((4,), S.u32),
                              vindex: S.i32,
                              soffset: S.i32,
                              aux: S.i32):
    # Load a vector and read individual elements
    vector4 = S.amdgpu.raw_buffer_load_x4(rsrc, vindex, soffset, aux)
    element0 = vector4[0]
    element1 = vector4[1]
    element3 = vector4[3]
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRVectorSubscriptWrite) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def vector_subscript_write_test(rsrc: S.Tensor((4,), S.u32),
                               vindex: S.i32,
                               soffset: S.i32,
                               aux: S.i32):
    # Load a vector and modify individual elements
    vector4 = S.amdgpu.raw_buffer_load_x4(rsrc, vindex, soffset, aux)
    vector4[0] = 42
    vector4[1] = vector4[2]
    vector4[3] = vindex
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRVectorSubscriptMixed) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def vector_subscript_mixed_test(rsrc: S.Tensor((4,), S.u32),
                               vindex: S.i32,
                               soffset: S.i32,
                               aux: S.i32):
    # Test mixed vector operations and subscripting
    vector4 = S.amdgpu.raw_buffer_load_x4(rsrc, vindex, soffset, aux)
    vector2 = S.amdgpu.raw_buffer_load_x2(rsrc, vindex, soffset, aux)

    # Read elements
    first_element = vector4[0]
    second_element = vector2[1]

    # Write elements
    vector4[2] = first_element
    vector2[0] = second_element
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRPrintf) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def printf_test(a: S.i32, b: S.f32):
    S.printf("Hello from GPU: %d %f\n", a, b)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRPrintfSimple) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def printf_simple():
    S.printf("Hello World!\n")
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRMemrefSlice) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def memref_slice(x: S.Tensor((8, 8), S.i32)):
    x_slice_1 = S.subview(x, (1, 1), (4, 4), (2, 2))
    a = x_slice_1[1, 1]
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRMemrefSliceUnBoundError) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def memref_slice(x: S.Tensor((8, 8), S.i32)):
    x_slice_1 = S.subview(x, (1, 2, 1), (4, 4, 5), (2, 2, 3))
    a = x_slice_1[1, 1]
)""""";

    RunMLIRGenerationErrorTest(kSourceCode,
                               "offsets, sizes and strides tuple sizes must "
                               "match memref rank");
}

TEST_F(MLIRGeneratorTest, GenerateMLIRMemrefSliceUnMatchError) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def memref_slice(x: S.Tensor((8, 8), S.i32)):
    x_slice_1 = S.subview(x, (1, 1), (4, 4, 5), (2, 3))
    a = x_slice_1[1, 1]
)""""";

    RunMLIRGenerationErrorTest(kSourceCode,
                               "offsets, sizes and strides must have the same "
                               "number of dimensions in subview() function");
}

TEST_F(MLIRGeneratorTest, GenerateMLIRMakeAndCastLayoutBasic) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def make_layout_test(x: S.Tensor((16,), S.f32)):
    # Create a simple 2D layout with dims (4, 4) and stride (4, 1)
    layout_1 = S.make_layout((4, 4), (4, 1))
    x_1 = S.view(x, S.f32, layout_1)
    x_1[2, 3] = S.convert(1.0, S.f32)
    # Create a CuTe-style nested layout: dims ((2, 4), 4) with stride ((4, 1), 8)
    layout_2 = S.make_layout(((2, 4), 4), ((4, 1), 8))
    x_2 = S.view(x, S.f32, layout_2)
    x_2[0, 0, 0] = S.convert(2.0, S.f32)
    # Linear index into nested dims should be expanded like CuTe.
    x_2[5, 0] = S.convert(3.0, S.f32)
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRMakeAndCastLayoutDynamic) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S
@avelang.jit
def make_layout_dynamic_test(x: S.Tensor((16,), S.f32), m: S.i32):
    # Create a layout with dynamic stride values
    layout_dyn = S.make_layout((4, 4), (m, 1))
    x_1 = S.view(x, S.f32, layout_dyn)
    # Mixed static and dynamic
    layout_mixed = S.make_layout((4, m), (m, 1))
)""""";
    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRMakeLayoutComplex) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def make_layout_complex_test():
    # Create multiple layouts with different shapes
    layout_1d = S.make_layout((16,), (1,))
    layout_2d = S.make_layout((4, 8), (8, 1))
    layout_3d = S.make_layout((2, 4, 8), (32, 8, 1))
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRMakeLayoutErrorMismatchedDims) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def make_layout_error_test():
    # Mismatched number of dims and stride elements
    layout = S.make_layout((4, 4), (4,))
)""""";

    RunMLIRGenerationErrorTest(
        kSourceCode,
        "Dims and stride tuples must have the same number of elements");
}

TEST_F(MLIRGeneratorTest, GenerateMLIRMakeLayoutErrorEmpty) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def make_layout_empty_test():
    # Empty dims and stride tuples
    layout = S.make_layout((), ())
)""""";

    RunMLIRGenerationErrorTest(kSourceCode,
                               "Empty tuple expressions are not supported");
}

TEST_F(MLIRGeneratorTest, GenerateMLIRLayoutAlgebraFunctions) {
    static const std::string kSourceCode = R"""""(
import substrate
import substrate.language as S

@substrate.jit
def layout_algebra_test(
    x8: S.Tensor((8,), S.i32),
    x24: S.Tensor((24,), S.i32),
    x32: S.Tensor((32,), S.i32),
    x48: S.Tensor((48,), S.i32),
):
    composition_layout = S.composition(
        S.make_layout((6, 2), (8, 2)),
        S.make_layout((4, 3), (3, 1)),
    )
    coalesced_layout = S.coalesce(S.make_layout((2, 4), (1, 2)))
    complement_layout = S.complement(S.make_layout((2, 2), (1, 6)), 24)
    product_layout = S.product(
        S.make_layout((2, 2), (4, 1)),
        S.make_layout((4, 2), (2, 1)),
    )
    divide_layout = S.divide(
        S.make_layout((4, 2, 3), (2, 1, 8)),
        S.make_layout((4,), (2,)),
    )

    x_comp = S.view(x48, S.i32, composition_layout)
    x_coal = S.view(x8, S.i32, coalesced_layout)
    x_cpl = S.view(x24, S.i32, complement_layout)
    x_prod = S.view(x32, S.i32, product_layout)
    x_div = S.view(x24, S.i32, divide_layout)

    x_comp[0, 0, 0] = x_coal[0]
    x_prod[0, 0, 0, 0] = x_div[0, 0, 0, 0]
    x_cpl[0, 0] = x_comp[1, 1, 2]
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRWithConstexpr) {
    static const std::string kSourceCode = R""""(
import avelang
import avelang.language as S

@avelang.jit
def kernel(N: S.constexpr, data: S.Tensor((16,), S.i32)):
    x = N + 1
    data[0] = x
)"""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();
    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);

    // Create the module first
    auto module = generator.CreateModule();

    // Now create constexpr values in that module
    mlir::OpBuilder builder(ir_context->GetMLIRContext());
    builder.setInsertionPointToStart(module.getBody());

    auto constAttr = builder.getI32IntegerAttr(32);
    auto constValue = mlir::arith::ConstantOp::create(
        builder, builder.getUnknownLoc(), constAttr);

    // Add to global frame
    auto *symbol_table = generator.GetSymbolTable();
    auto &global_frame = symbol_table->GetCurrentFrame();
    global_frame.AddValue("N", constValue, /*immutable=*/true);

    // Generate the MLIR from the AST
    module = generator.Generate(root);

    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

    // Verify the function signature excludes the constexpr parameter
    mlir::func::FuncOp func;
    module.walk([&func](mlir::func::FuncOp f) {
        if (f.getName() == "kernel") {
            func = f;
        }
    });
    ASSERT_TRUE(func);

    // Should have 1 argument (data memref), not 2 (N is constexpr, excluded)
    EXPECT_EQ(func.getNumArguments(), 1u);
}

TEST_F(MLIRGeneratorTest, ConstexprRedefinitionError) {
    static const std::string kSourceCode = R""""(
import avelang
import avelang.language as S

@avelang.jit
def kernel(N: S.constexpr):
    N = 10
)"""";

    ast::ASTNode *root;
    TryParse(kSourceCode, &root);
    ASSERT_NE(root, nullptr);

    auto ir_context = ir::IRContext::Create();
    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);

    // Create the module first
    auto module = generator.CreateModule();

    // Now create constexpr values in that module
    mlir::OpBuilder builder(ir_context->GetMLIRContext());
    builder.setInsertionPointToStart(module.getBody());

    auto constAttr = builder.getI32IntegerAttr(32);
    auto constValue = mlir::arith::ConstantOp::create(
        builder, builder.getUnknownLoc(), constAttr);

    // Add to global frame
    auto *symbol_table = generator.GetSymbolTable();
    auto &global_frame = symbol_table->GetCurrentFrame();
    global_frame.AddValue("N", constValue, /*immutable=*/true);

    // Generate the MLIR from the AST - should fail with immutable error
    module = generator.Generate(root);

    // Should fail with error about immutable assignment
    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    std::string errors = diagHandler_.GetErrorMessages(&SM);
    EXPECT_TRUE(errors.find("Cannot assign to immutable") !=
                    std::string::npos ||
                errors.find("constexpr") != std::string::npos);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRFull) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def full_test(data: S.Tensor((16,), S.i32)):
    ones = S.full((4,), 1, S.i32)
    twos = S.full((2, 3), 2, S.i32)
    data[0] = ones[2] + twos[1, 2]
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRViewFromTensor) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def view_tensor(data: S.Tensor((16,), S.i32), vector: S.Tensor((4,), S.u32)):
    data_casted = S.view(data, S.Tensor((4, 4), S.u32))
    vector = data_casted[0]
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRViewFromValue) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def view_tensor(data: S.Tensor((16, 2), S.u32), vector: S.Tensor((4,), S.u32)):
    vector_casted = S.view(vector, S.Tensor((2, 2), S.u32))
    data[0] = vector_casted[0]
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRViewFromPackedSubscriptedValue) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def view_packed_subscript(shm: S.Tensor((4096,), S.u32), out: S.Tensor((8,), S.bf16)):
    packed = S.view(shm, S.Tensor((128, 8, 4), S.u32))
    frag = S.view(packed[0, 0], S.Tensor((2, 4, 1), S.bf16))
    for i in S.range(2):
        for j in S.range(4):
            out[i * 4 + j] = frag[i, j, 0]
)""""";

    RunMLIRGenerationTest(kSourceCode);
}

TEST_F(MLIRGeneratorTest, GenerateMLIRCallUsesSubviewForTensorSubscriptArg) {
    static const std::string kCallerSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def call_with_subscript_arg():
    buf = S.make_local((2, 4), S.u32)
    write_lane(buf[0])
)""""";

    static const std::string kHelperSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def write_lane(dst: S.Tensor((4,), S.u32)):
    dst[0] = 7
)""""";

    ast::ASTNode *callerRoot;
    TryParse(kCallerSourceCode, &callerRoot);
    ASSERT_NE(callerRoot, nullptr);

    ast::ASTNode *helperRoot;
    TryParse(kHelperSourceCode, &helperRoot);
    ASSERT_NE(helperRoot, nullptr);
    auto *helperModule = llvm::dyn_cast<ast::Module>(helperRoot);
    ASSERT_NE(helperModule, nullptr);
    ast::FunctionDef *helperFunc = nullptr;
    for (auto *stmt : helperModule->GetBody()) {
        helperFunc = llvm::dyn_cast<ast::FunctionDef>(stmt);
        if (helperFunc) {
            break;
        }
    }
    ASSERT_NE(helperFunc, nullptr);

    auto ir_context = ir::IRContext::Create();

    ir::MLIRGenerator generator(ir_context.get(), diagnostics_);
    generator.RegisterJitDependency(helperFunc);
    auto mlir = generator.Generate(callerRoot);

    const clang::SourceManager &SM = diagnostics_->GetSourceManager();
    ASSERT_FALSE(diagnostics_->GetEngine()->hasErrorOccurred())
        << diagHandler_.GetErrorMessages(&SM);
    ASSERT_NE(mlir, nullptr);

    bool foundCall = false;
    mlir->walk([&](mlir::func::CallOp op) {
        if (!op.getCallee().contains("write_lane")) {
            return;
        }
        foundCall = true;
        ASSERT_EQ(op.getNumOperands(), 1u);
        auto arg = op.getOperand(0);
        EXPECT_TRUE(mlir::isa<cf::MemRefType>(arg.getType()));
        EXPECT_TRUE(arg.getDefiningOp<cf::AveLangMemRefSubViewOp>());
        EXPECT_FALSE(arg.getDefiningOp<cf::AveLangMemRefAllocaOp>());
    });

    EXPECT_TRUE(foundCall);

    mlir::PassManager pm(mlir.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(mlir))) << "Pass pipeline failed";

    ASSERT_TRUE(mlir::succeeded(mlir::verify(mlir)))
        << "MLIR verification failed!";
}

TEST_F(MLIRGeneratorTest, GenerateMLIRViewWithMismatchSize) {
    static const std::string kSourceCode = R"""""(
import avelang
import avelang.language as S

@avelang.jit
def view_tensor(data: S.Tensor((16,), S.i32), vector: S.Tensor((4,), S.u32)):
    data_casted = S.view(data, S.Tensor((16, 4), S.u32))
    vector = data_casted[0]
)""""";

    RunMLIRGenerationErrorTest(
        kSourceCode,
        "The source memref and target type must have the same total byte size");
}

} // namespace causalflow::avelang::frontend
