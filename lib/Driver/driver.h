#pragma once

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace llvm {
class LLVMContext;
}

namespace mlir {
class ModuleOp;
}

namespace causalflow::avelang {

namespace ast {
class ASTNode;
}

namespace basic {
class DiagnosticManager;
}

namespace driver {

/// Compilation output stage enum
enum class OutputStage { AST, MLIR, LLVMIR, Binary };

/// Compilation options following LLVM style.
struct CompilationOptions {
    /// Target GPU chipset (default: gfx90a)
    llvm::StringRef TargetChipset = "gfx90a";

    /// Target triple (default: amdgcn-amd-amdhsa)
    llvm::StringRef TargetTriple = "amdgcn-amd-amdhsa";

    /// Optimization level (0-3, default: 2)
    unsigned OptLevel = 2;

    /// Number of warps (or waves) requested by the frontend. -1 means unset.
    int NumWarps = -1;

    /// Use bare pointer calling convention for memref
    bool UseBarePointerCallConv = true;

    /// Output stage (default: LLVMIR)
    OutputStage Stage = OutputStage::LLVMIR;
};

/// AveLang compilation driver following LLVM/Clang patterns.
class Driver {
  public:
    Driver(llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> DiagMgr,
           ::llvm::LLVMContext &LLVMCtx);
    ~Driver();

    /// Main compilation entry point from file.
    llvm::Error compileFromFile(llvm::StringRef InputFile,
                                llvm::raw_ostream &OS,
                                const CompilationOptions &Options = {});

    /// Main compilation entry point from memory buffer.
    llvm::Error compileFromBuffer(const llvm::MemoryBuffer &InputBuffer,
                                  llvm::raw_ostream &OS,
                                  const CompilationOptions &Options = {},
                                  llvm::StringRef BufferName = "<input>");

    /// Compile with JSON constexprs parsed into canonical typed attributes.
    llvm::Error compileFromBufferWithConstexprs(
        const llvm::MemoryBuffer &InputBuffer, llvm::raw_ostream &OS,
        const CompilationOptions &Options, llvm::StringRef BufferName,
        llvm::StringRef ConstexprsJSON);

    /// Individual compilation stages for testing.
    llvm::Expected<ast::ASTNode *>
    parseBuffer(const llvm::MemoryBuffer &Buffer,
                llvm::StringRef BufferName = "<input>");

    llvm::Expected<mlir::ModuleOp> generateMLIR(ast::ASTNode *AST);

    llvm::Expected<std::unique_ptr<::llvm::Module>>
    generateLLVMIR(mlir::ModuleOp MLIRModule,
                   const CompilationOptions &Options = {});

    llvm::Expected<std::string>
    generateBinary(mlir::ModuleOp MLIRModule,
                   const CompilationOptions &Options = {});

    /// Output helpers.
    llvm::Error emitAST(ast::ASTNode *AST, llvm::raw_ostream &OS);
    llvm::Error emitMLIR(mlir::ModuleOp MLIRModule, llvm::raw_ostream &OS);
    llvm::Error emitLLVMIR(::llvm::Module &LLVMModule, llvm::raw_ostream &OS);
    llvm::Error emitBinary(llvm::StringRef BinaryCode, llvm::raw_ostream &OS);

  private:
    class Impl;
    std::unique_ptr<Impl> Impl;
};

} // namespace driver
} // namespace causalflow::avelang
