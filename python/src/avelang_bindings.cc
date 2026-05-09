#include "AST/ast.h"
#include "AST/ast_context.h"
#include "AST/ast_nodes_stmt.h"
#include "Basic/diagnostic.h"
#include "Driver/driver.h"
#include "Frontend/avelang_parser.h"
#include "IR/ir_context.h"
#include "IR/mlir_generator.h"
#include "Target/GPU/gpu_backend.h"
#include "Target/GPU/lower_to_llvm.h"

#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <vector>

namespace py = pybind11;
using namespace llvm;
using namespace causalflow::avelang;

static ir::MLIRGenerator::FunctionType
parseFunctionType(const std::string &function_type) {
    if (function_type == "kernel" || function_type == "global_kernel") {
        return ir::MLIRGenerator::FunctionType::kGlobalKernel;
    }
    if (function_type == "jit") {
        return ir::MLIRGenerator::FunctionType::kPrivateFunction;
    }
    if (function_type == "host") {
        return ir::MLIRGenerator::FunctionType::kHostFunction;
    }
    throw std::runtime_error("Unknown function_type: " + function_type);
}

static std::string NormalizeTargetTriple(llvm::StringRef triple) {
    if (triple.empty()) {
        return triple.str();
    }
    std::string lower = triple.lower();
    if (lower == "amdgpu" || lower == "amdgcn" || lower == "rocm" ||
        lower == "hip") {
        return "amdgcn-amd-amdhsa";
    }
    return triple.str();
}

static std::string FormatMLIRLocation(mlir::Location loc) {
    std::string location;
    llvm::raw_string_ostream os(location);
    loc.print(os);
    os.flush();
    return location;
}

static bool HasMeaningfulMLIRLocation(llvm::StringRef location) {
    return !location.empty() && location != "loc(unknown)";
}

static std::string
DescribeRemainingTranslationBlockers(mlir::Operation *operation) {
    if (!operation) {
        return {};
    }

    std::set<std::string> blockers;
    operation->walk([&](mlir::UnrealizedConversionCastOp cast) {
        std::string location = FormatMLIRLocation(cast.getLoc());
        if (!HasMeaningfulMLIRLocation(location)) {
            return;
        }
        std::string detail = location + ": builtin.unrealized_conversion_cast";
        detail += " (";
        if (!cast.getInputs().empty()) {
            detail += "from ";
            {
                std::string fromTy;
                llvm::raw_string_ostream os(fromTy);
                cast.getInputs().front().getType().print(os);
                os.flush();
                detail += fromTy;
            }
        }
        if (!cast.getOutputs().empty()) {
            if (!cast.getInputs().empty()) {
                detail += " ";
            }
            detail += "to ";
            {
                std::string toTy;
                llvm::raw_string_ostream os(toTy);
                cast.getOutputs().front().getType().print(os);
                os.flush();
                detail += toTy;
            }
        }
        detail += ")";
        blockers.insert(std::move(detail));
    });

    if (blockers.empty()) {
        return {};
    }

    std::string message = "remaining translation blockers:\n";
    for (const auto &blocker : blockers) {
        message += "  " + blocker + "\n";
    }
    return message;
}

class AveLangCompiler {
  public:
    AveLangCompiler()
        : diag_opts_(std::make_unique<clang::DiagnosticOptions>()),
          diag_consumer_(errs(), *diag_opts_),
          diag_manager_(new basic::DiagnosticManager(&diag_consumer_)),
          driver_(diag_manager_, llvm_context_) {}

    std::string
    compileToBinary(const std::string &source_code,
                    const std::string &target_triple = "amdgcn-amd-amdhsa",
                    const std::string &target_chipset = "gfx90a",
                    unsigned opt_level = 2, int num_warps = -1,
                    const std::string &constexprs_json = "[]") {

        // Create compilation options
        causalflow::avelang::driver::CompilationOptions options;
        options.TargetTriple = target_triple;
        options.TargetChipset = target_chipset;
        options.OptLevel = opt_level;
        options.NumWarps = num_warps;
        options.Stage = causalflow::avelang::driver::OutputStage::Binary;

        // Create memory buffer from source
        auto buffer =
            llvm::MemoryBuffer::getMemBuffer(source_code, "<jit_input>");

        // Compile to binary with constexprs
        std::string output;
        llvm::raw_string_ostream os(output);

        auto error = driver_.compileFromBufferWithConstexprs(
            *buffer, os, options, "<jit_input>", constexprs_json);
        if (error) {
            throw std::runtime_error("Compilation failed: " +
                                     llvm::toString(std::move(error)));
        }

        return output;
    }

    py::bytes compileToBinaryBytes(const std::string &source_code,
                                   const std::string &target_triple,
                                   const std::string &target_chipset,
                                   unsigned opt_level = 2, int num_warps = -1,
                                   const std::string &constexprs_json = "[]") {
        std::string binary =
            compileToBinary(source_code, target_triple, target_chipset,
                            opt_level, num_warps, constexprs_json);
        return py::bytes(binary);
    }

  private:
    LLVMContext llvm_context_;
    std::unique_ptr<clang::DiagnosticOptions> diag_opts_;
    clang::TextDiagnosticPrinter diag_consumer_;
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diag_manager_;
    driver::Driver driver_;
};

class AveLangMLIRGenerator {
  public:
    AveLangMLIRGenerator();
    void CreateModule();
    void Generate(ast::ASTNode *root);
    void InjectConstexprs(const std::string &constexprs_json);
    void VisitFunctionDef(const py::object &py_function_def,
                          const std::string &constexprs_json = "[]",
                          const std::string &function_type = "kernel");
    void GenerateFromPythonAST(const py::object &py_ast_node);
    void AddJitDependency(const py::object &py_ast_node);
    std::string GetMLIR();
    std::string GetLLVMIR(const std::string &target_triple,
                          const std::string &target_chipset, unsigned opt_level,
                          int num_warps = -1);
    std::string GetAssembly(const std::string &target_triple,
                            const std::string &target_chipset,
                            unsigned opt_level, int num_warps = -1);
    py::bytes CompileToBinaryBytes(const std::string &target_triple,
                                   const std::string &target_chipset,
                                   unsigned opt_level, int num_warps = -1);

  private:
    void ClearMLIRDiagnostics();
    std::string FormatLoweringFailure(mlir::Operation *operation) const;
    void ThrowIfDiagnosticsFailed(const std::string &context);

    LLVMContext llvm_context_;
    std::unique_ptr<clang::DiagnosticOptions> diag_opts_;
    clang::TextDiagnosticPrinter diag_consumer_;
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diag_manager_;
    std::unique_ptr<ir::IRContext> ir_context_;
    llvm::SmallVector<std::string, 4> mlir_errors_;
    std::vector<llvm::IntrusiveRefCntPtr<ast::ASTContext>> dependency_contexts_;
    std::vector<std::unique_ptr<llvm::MemoryBuffer>> owned_buffers_;
    ir::MLIRGenerator generator_;
};

AveLangMLIRGenerator::AveLangMLIRGenerator()
    : diag_opts_(std::make_unique<clang::DiagnosticOptions>()),
      diag_consumer_(errs(), *diag_opts_),
      diag_manager_(new basic::DiagnosticManager(&diag_consumer_)),
      ir_context_(ir::IRContext::Create()),
      generator_(ir_context_.get(), diag_manager_) {
    ir_context_->GetMLIRContext()->getDiagEngine().registerHandler(
        [this](mlir::Diagnostic &diag) {
            if (diag.getSeverity() != mlir::DiagnosticSeverity::Error) {
                return mlir::success();
            }

            std::string message = diag.str();
            std::string location = FormatMLIRLocation(diag.getLocation());
            if (HasMeaningfulMLIRLocation(location)) {
                message = location + ": " + message;
            }
            mlir_errors_.push_back(std::move(message));
            return mlir::success();
        });
}

void AveLangMLIRGenerator::ClearMLIRDiagnostics() { mlir_errors_.clear(); }

std::string
AveLangMLIRGenerator::FormatLoweringFailure(mlir::Operation *operation) const {
    std::string message;
    for (const auto &error : mlir_errors_) {
        message += error + "\n";
    }

    std::string blockers = DescribeRemainingTranslationBlockers(operation);
    if (!blockers.empty()) {
        if (!message.empty()) {
            message += "\n";
        }
        message += blockers;
    }

    if (!message.empty() && message.back() == '\n') {
        message.pop_back();
    }
    return message;
}

void AveLangMLIRGenerator::ThrowIfDiagnosticsFailed(
    const std::string &context) {
    if (diag_manager_ && diag_manager_->GetEngine() &&
        diag_manager_->GetEngine()->hasErrorOccurred()) {
        throw std::runtime_error(context);
    }
}

void AveLangMLIRGenerator::CreateModule() { generator_.CreateModule(); }

void AveLangMLIRGenerator::Generate(ast::ASTNode *root) {
    if (!root) {
        throw std::runtime_error("AST root is null");
    }
    auto module = generator_.Generate(root);
    ThrowIfDiagnosticsFailed("Failed to generate MLIR from AST");
    if (!module) {
        throw std::runtime_error("Failed to generate MLIR from AST");
    }
}

void AveLangMLIRGenerator::InjectConstexprs(
    const std::string &constexprs_json) {
    auto error = generator_.InjectConstexprs(constexprs_json);
    if (error) {
        throw std::runtime_error("Constexpr injection failed: " +
                                 llvm::toString(std::move(error)));
    }
}

void AveLangMLIRGenerator::VisitFunctionDef(const py::object &py_function_def,
                                            const std::string &constexprs_json,
                                            const std::string &function_type) {
    llvm::IntrusiveRefCntPtr<ast::ASTContext> ast_context(
        new ast::ASTContext());
    frontend::AveLangParser parser(ast_context, diag_manager_);
    ast::FunctionDef *func =
        parser.ParsePythonFunctionDef(py_function_def, "<jit_input>");
    if (!func) {
        throw std::runtime_error("Failed to parse Python FunctionDef");
    }
    if (auto buffer = parser.ReleaseOwnedBuffer()) {
        owned_buffers_.push_back(std::move(buffer));
    }
    ThrowIfDiagnosticsFailed("Failed to parse Python FunctionDef");
    auto error = generator_.VisitFunctionDefWithType(
        func, parseFunctionType(function_type), constexprs_json);
    if (error) {
        throw std::runtime_error("VisitFunctionDef failed: " +
                                 llvm::toString(std::move(error)));
    }
    ThrowIfDiagnosticsFailed(
        "VisitFunctionDef failed due to compiler diagnostics");
}

void AveLangMLIRGenerator::GenerateFromPythonAST(
    const py::object &py_ast_node) {
    llvm::IntrusiveRefCntPtr<ast::ASTContext> ast_context(
        new ast::ASTContext());
    frontend::AveLangParser parser(ast_context, diag_manager_);
    ast::ASTNode *root = parser.ParsePythonAST(py_ast_node, "<jit_input>");
    if (!root) {
        throw std::runtime_error("Failed to parse Python AST module");
    }
    if (auto buffer = parser.ReleaseOwnedBuffer()) {
        owned_buffers_.push_back(std::move(buffer));
    }
    auto module = generator_.Generate(root);
    ThrowIfDiagnosticsFailed("Failed to generate MLIR from Python AST");
    if (!module) {
        throw std::runtime_error("Failed to generate MLIR from AST");
    }
}

void AveLangMLIRGenerator::AddJitDependency(const py::object &py_ast_node) {
    llvm::IntrusiveRefCntPtr<ast::ASTContext> ast_context(
        new ast::ASTContext());
    frontend::AveLangParser parser(ast_context, diag_manager_);
    ast::ASTNode *root = parser.ParsePythonAST(py_ast_node, "<jit_input>");
    if (!root) {
        throw std::runtime_error("Failed to parse Python AST module");
    }
    if (auto buffer = parser.ReleaseOwnedBuffer()) {
        owned_buffers_.push_back(std::move(buffer));
    }
    ThrowIfDiagnosticsFailed("Failed to parse Python AST for JIT dependency");
    auto *module = llvm::dyn_cast<ast::Module>(root);
    if (!module) {
        throw std::runtime_error("Expected AST Module for JIT dependency");
    }
    for (auto *stmt : module->GetBody()) {
        if (auto *func = llvm::dyn_cast<ast::FunctionDef>(stmt)) {
            generator_.RegisterJitDependency(func);
        }
    }
    dependency_contexts_.push_back(std::move(ast_context));
}

std::string AveLangMLIRGenerator::GetMLIR() {
    auto module = generator_.GetModule();
    std::string output;
    llvm::raw_string_ostream os(output);
    module.print(os);
    return os.str();
}

std::string AveLangMLIRGenerator::GetLLVMIR(const std::string &target_triple,
                                            const std::string &target_chipset,
                                            unsigned opt_level, int num_warps) {
    auto module = generator_.GetModule();
    ThrowIfDiagnosticsFailed(
        "Failed to lower MLIR to LLVM IR due to compiler diagnostics");
    if (!module) {
        throw std::runtime_error("MLIR module is null");
    }

    std::string normalized_triple = NormalizeTargetTriple(target_triple);
    target::gpu::GPUCompilationOptions options;
    options.triple = normalized_triple;
    options.chipset = target_chipset;
    options.optimization_level = opt_level;
    options.num_warps = num_warps;

    auto backend =
        target::gpu::GPUBackendRegistry::getInstance().createBackendForTriple(
            options.triple);
    if (!backend) {
        throw std::runtime_error("No backend available for target: " +
                                 target_triple);
    }
    backend->EnsureInitialized();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

    target::gpu::LowerToLLVM lowerer(ir_context_.get());
    ClearMLIRDiagnostics();
    auto llvm_module = lowerer.compile(module, llvm_context_, options);
    if (!llvm_module) {
        std::string detail = FormatLoweringFailure(module);
        if (!detail.empty()) {
            throw std::runtime_error("Failed to lower MLIR to LLVM IR:\n" +
                                     detail);
        }
        throw std::runtime_error("Failed to lower MLIR to LLVM IR");
    }

    std::string output;
    llvm::raw_string_ostream os(output);
    llvm_module->print(os, nullptr);
    return os.str();
}

std::string AveLangMLIRGenerator::GetAssembly(const std::string &target_triple,
                                              const std::string &target_chipset,
                                              unsigned opt_level,
                                              int num_warps) {
    auto module = generator_.GetModule();
    ThrowIfDiagnosticsFailed(
        "Failed to lower MLIR to LLVM IR due to compiler diagnostics");
    if (!module) {
        throw std::runtime_error("MLIR module is null");
    }

    std::string normalized_triple = NormalizeTargetTriple(target_triple);
    target::gpu::GPUCompilationOptions options;
    options.triple = normalized_triple;
    options.chipset = target_chipset;
    options.optimization_level = opt_level;
    options.num_warps = num_warps;

    auto backend =
        target::gpu::GPUBackendRegistry::getInstance().createBackendForTriple(
            options.triple);
    if (!backend) {
        throw std::runtime_error("No backend available for target: " +
                                 target_triple);
    }
    backend->EnsureInitialized();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

    target::gpu::LowerToLLVM lowerer(ir_context_.get());
    ClearMLIRDiagnostics();
    auto llvm_module = lowerer.compile(module, llvm_context_, options);
    if (!llvm_module) {
        std::string detail = FormatLoweringFailure(module);
        if (!detail.empty()) {
            throw std::runtime_error("Failed to lower MLIR to LLVM IR:\n" +
                                     detail);
        }
        throw std::runtime_error("Failed to lower MLIR to LLVM IR");
    }

    auto assembly_or_error = backend->generateAssembly(*llvm_module, options);
    if (!assembly_or_error) {
        throw std::runtime_error("Failed to generate assembly: " +
                                 llvm::toString(assembly_or_error.takeError()));
    }

    return *assembly_or_error;
}

py::bytes
AveLangMLIRGenerator::CompileToBinaryBytes(const std::string &target_triple,
                                           const std::string &target_chipset,
                                           unsigned opt_level, int num_warps) {
    auto module = generator_.GetModule();
    ThrowIfDiagnosticsFailed(
        "Failed to lower MLIR to LLVM IR due to compiler diagnostics");
    if (!module) {
        throw std::runtime_error("MLIR module is null");
    }

    std::string normalized_triple = NormalizeTargetTriple(target_triple);
    target::gpu::GPUCompilationOptions options;
    options.triple = normalized_triple;
    options.chipset = target_chipset;
    options.optimization_level = opt_level;
    options.num_warps = num_warps;

    auto backend =
        target::gpu::GPUBackendRegistry::getInstance().createBackendForTriple(
            options.triple);
    if (!backend) {
        throw std::runtime_error("No backend available for target: " +
                                 target_triple);
    }
    backend->EnsureInitialized();

    target::gpu::LowerToLLVM lowerer(ir_context_.get());
    ClearMLIRDiagnostics();
    auto llvm_module = lowerer.compile(module, llvm_context_, options);
    if (!llvm_module) {
        std::string detail = FormatLoweringFailure(module);
        if (!detail.empty()) {
            throw std::runtime_error("Failed to lower MLIR to LLVM IR:\n" +
                                     detail);
        }
        throw std::runtime_error("Failed to lower MLIR to LLVM IR");
    }

    auto binary_or_error = backend->generateBinary(*llvm_module, options);
    if (!binary_or_error) {
        throw std::runtime_error("Failed to generate binary: " +
                                 llvm::toString(binary_or_error.takeError()));
    }
    return py::bytes(*binary_or_error);
}

PYBIND11_MODULE(_avelang_bindings, m) {
    m.doc() = "ave-lang JIT compiler Python bindings";

    py::class_<AveLangCompiler>(m, "AveLangCompiler")
        .def(py::init<>())
        .def("compile_to_binary_bytes", &AveLangCompiler::compileToBinaryBytes,
             "Compile avelang source to binary bytes", py::arg("source_code"),
             py::arg("target_triple"), py::arg("target_chipset"),
             py::arg("opt_level") = 2, py::arg("num_warps") = -1,
             py::arg("constexprs_json") = "[]");

    py::class_<AveLangMLIRGenerator>(m, "MLIRGenerator")
        .def(py::init<>())
        .def("create_module", &AveLangMLIRGenerator::CreateModule)
        .def("generate", &AveLangMLIRGenerator::Generate, py::arg("root"))
        .def("inject_constexprs", &AveLangMLIRGenerator::InjectConstexprs,
             py::arg("constexprs_json") = "[]")
        .def("visit_function_def", &AveLangMLIRGenerator::VisitFunctionDef,
             py::arg("py_function_def"), py::arg("constexprs_json") = "[]",
             py::arg("function_type") = "kernel")
        .def("generate_from_python_ast",
             &AveLangMLIRGenerator::GenerateFromPythonAST,
             py::arg("py_ast_node"))
        .def("add_jit_dependency", &AveLangMLIRGenerator::AddJitDependency,
             py::arg("py_ast_node"))
        .def("get_mlir", &AveLangMLIRGenerator::GetMLIR)
        .def("get_llvm_ir", &AveLangMLIRGenerator::GetLLVMIR,
             py::arg("target_triple"), py::arg("target_chipset"),
             py::arg("opt_level") = 2, py::arg("num_warps") = -1)
        .def("get_assembly", &AveLangMLIRGenerator::GetAssembly,
             py::arg("target_triple"), py::arg("target_chipset"),
             py::arg("opt_level") = 2, py::arg("num_warps") = -1)
        .def("compile_to_binary_bytes",
             &AveLangMLIRGenerator::CompileToBinaryBytes,
             py::arg("target_triple"), py::arg("target_chipset"),
             py::arg("opt_level"), py::arg("num_warps") = -1);
}
