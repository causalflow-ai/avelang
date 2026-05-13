//===- AveLangOps.td - AveLang dialect ops ----------------*- tablegen -*-===//
//
// This file defines the operations of the AveLang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AVE_LANG_OPS
#define AVE_LANG_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AveLang Dialect
//===----------------------------------------------------------------------===//

def AveLang_Dialect : Dialect {
    let name = "ave";
    let cppNamespace = "::causalflow::avelang::dialect";
    let summary = "AveLang dialect";
    let description = [{
        The AveLang dialect provides operations for avelang
        computations, including tuple operations and specialized kernels.
    }];

    let extraClassDeclaration = [{
        void registerTypes();

        ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
        void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
    }];
}

//===----------------------------------------------------------------------===//
// Base operation classes
//===----------------------------------------------------------------------===//

class AveLang_Op<string mnemonic, list<Trait> traits = []> :
    Op<AveLang_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AveLang Operations
//===----------------------------------------------------------------------===//

def ReturnOp : AveLang_Op<"return"> {
    let summary = "Return from an ave-lang function";
    let description = [{
        The `ave.return` operation represents an early return before
        control-flow normalization rewrites functions to a single-exit form.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);
    let results = (outs);
}

def MakeIntTupleOp : AveLang_Op<"make_int_tuple", [Pure]> {
    let summary = "Create an integer tuple from elements";
    let description = [{
        The `ave.make_int_tuple` operation creates a tuple containing
        integer or index values, or nested tuples. This replaces the 
        ParseNestedTuple functionality from builtin_module.cc.
        
        Example:
        ```mlir
        %tuple = ave.make_int_tuple %0, %1, %2 : i32, i32, i32
        %nested = ave.make_int_tuple %tuple, %3 : none, i32
        ```
    }];
    
    let arguments = (ins Variadic<AnyType>:$elements);
    let results = (outs AnyType:$result);
    
    let assemblyFormat = [{
        $elements attr-dict `:` type($elements) `->` type($result)
    }];
    
    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$elements)>
    ];
    
    let hasVerifier = 1;
    
    let extraClassDeclaration = [{
        // Helper methods for tuple access
        size_t getNumElements() { return getElements().size(); }
        mlir::Value getElement(size_t index) { 
            assert(index < getNumElements());
            return getElements()[index]; 
        }
    }];
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

include "AveLangMemRefOps.td"

def MakeLayoutOp : AveLang_Op<"make_layout", [Pure]> {
    let summary = "Create a layout from dims and stride tuples";
    let description = [{
        The `ave.make_layout` operation creates a Layout value from
        dims and stride integer tuples. The Layout represents memory layout
        information that can be used in tensor operations.

        Example:
        ```mlir
        %dims = ave.make_int_tuple %c4, %c4 : i32, i32 -> none
        %stride = ave.make_int_tuple %c4, %c1 : i32, i32 -> none
        %layout = ave.make_layout %dims, %stride : none, none -> none
        ```
    }];

    let arguments = (ins
        AnyType:$dims,
        AnyType:$stride
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $dims `,` $stride attr-dict `:`
        functional-type(operands, results)
    }];

    let builders = [
        OpBuilder<(ins "mlir::Value":$dims, "mlir::Value":$stride)>
    ];

    let hasVerifier = 1;
}

def FullOp : AveLang_Op<"full", [Pure]> {
    let summary = "Create a memref filled with a constant value";
    let description = [{
        The `ave.full` operation creates an ave-lang memref filled with a constant scalar value.
        The memref is flattened to a 1D buffer, filled using vector broadcast and transfer write,
        then reshaped to the target shape.

        Example:
        ```mlir
        %result = ave.full %shape, %value : i32 ->
            !ave.memref<!ave.layout<dims = [4, 4], strides = []>, i32>
        ```
    }];

    let arguments = (ins
        AnyType:$shape,
        AnyType:$value
    );
    let results = (outs AveLang_MemRef:$result);

    let assemblyFormat = [{
        $shape `,` $value attr-dict `:`
        type($shape) `,` type($value) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::Value":$shape, "mlir::Value":$value,
                      "mlir::Type":$resultType)>
    ];

    let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// GPU Operations
//===----------------------------------------------------------------------===//

// Base class for GPU operations
class AveLang_GPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<AveLang_Dialect, "gpu." # mnemonic, traits>;

//===----------------------------------------------------------------------===//
// NVVM Operations
//===----------------------------------------------------------------------===//

def NVVMMMAOp : AveLang_GPU_Op<"nvvm_mma", []> {
    let summary = "NVVM Matrix Multiply-Accumulate operation";
    let description = [{
        The `ave.gpu.nvvm_mma` operation performs matrix multiplication and accumulation
        using NVVM MMA intrinsics.
    }];

    let arguments = (ins
        LLVM_AnyVector:$a,
        LLVM_AnyVector:$b,
        LLVM_AnyVector:$c
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $a `,` $b `,` $c attr-dict `:`
        type($a) `,` type($b) `,` type($c) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def NVVMLdMatrixOp : AveLang_GPU_Op<"nvvm_ldmatrix", []> {
    let summary = "NVVM Load Matrix operation";
    let description = [{
        The `ave.gpu.nvvm_ldmatrix` operation loads a matrix from shared memory
        using NVVM LdMatrix intrinsics.
    }];

    let arguments = (ins
        AveLang_MemRef:$memref,
        StrAttr:$matrix_shape,
        I32Attr:$matrix_num,
        I32Attr:$matrix_bit_width
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $memref attr-dict `:` type($memref) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def NVVMStMatrixOp : AveLang_GPU_Op<"nvvm_stmatrix", []> {
    let summary = "NVVM Store Matrix operation";
    let description = [{
        The `ave.gpu.nvvm_stmatrix` operation stores a matrix to shared memory
        using NVVM StMatrix intrinsics.
    }];

    let arguments = (ins
        AnyType:$value,
        AveLang_MemRef:$memref,
        StrAttr:$matrix_shape,
        I32Attr:$matrix_num,
        I32Attr:$matrix_bit_width
    );
    let results = (outs);

    let assemblyFormat = [{
        $value `,` $memref attr-dict `:`
        type($value) `,` type($memref)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}


def NVVMTMADescriptorOp : AveLang_GPU_Op<"nvvm_tma_descriptor", []> {
    let summary = "NVVM TMA Descriptor operation";
    let description = [{
        The `ave.gpu.nvvm_tma_descriptor` operation creates an NVGPU
        tensor-map descriptor for Tensor Memory Access from a memref operand.
        Box dimensions are derived from the provided shared-memory layout.
    }];

    let arguments = (ins
        AveLang_MemRef:$memref,
        AnyType:$layout
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $memref `,` $layout attr-dict `:` type($memref) `,` type($layout)
        `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];
}
//===----------------------------------------------------------------------===//
// AMDGPU Operations
//===----------------------------------------------------------------------===//

def AMDGPUMfmaOp : AveLang_GPU_Op<"amdgpu_mfma", []> {
    let summary = "AMDGPU MFMA operation";
    let description = [{
        The `ave.gpu.amdgpu_mfma` operation performs matrix multiplication and accumulation
        using AMDGPU MFMA intrinsics. The operation is configured with matrix dimensions (m, n, k)
        and data types (type_a, type_c) to determine the appropriate intrinsic to use.
    }];

    let arguments = (ins
        LLVM_AnyVector:$a,
        LLVM_AnyVector:$b,
        LLVM_AnyVector:$c,
        I32Attr:$m,
        I32Attr:$n,
        I32Attr:$k,
        StrAttr:$type_a,
        StrAttr:$type_c
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $a `,` $b `,` $c attr-dict `:`
        type($a) `,` type($b) `,` type($c) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def AMDGPURawBufferLoadOp : AveLang_GPU_Op<"amdgpu_raw_buffer_load", []> {
    let summary = "AMDGPU Raw Buffer Load operation";
    let description = [{
        The `ave.gpu.amdgpu_raw_buffer_load` operation loads data from a raw buffer
        using AMDGPU raw buffer intrinsics.
    }];

    let arguments = (ins
        AnyType:$rsrc,
        AnyType:$vindex,
        AnyType:$soffset,
        AnyType:$aux
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $rsrc `,` $vindex `,` $soffset `,` $aux attr-dict `:`
        type($rsrc) `,` type($vindex) `,` type($soffset) `,` type($aux) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def AMDGPURawBufferStoreOp : AveLang_GPU_Op<"amdgpu_raw_buffer_store", []> {
    let summary = "AMDGPU Raw Buffer Store operation";
    let description = [{
        The `ave.gpu.amdgpu_raw_buffer_store` operation stores data to a raw buffer
        using AMDGPU raw buffer intrinsics.
    }];

    let arguments = (ins
        AnyType:$data,
        AnyType:$rsrc,
        AnyType:$vindex,
        AnyType:$soffset,
        AnyType:$aux
    );
    let results = (outs);

    let assemblyFormat = [{
        $data `,` $rsrc `,` $vindex `,` $soffset `,` $aux attr-dict `:`
        type($data) `,` type($rsrc) `,` type($vindex) `,` type($soffset) `,` type($aux)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

#endif // AVE_LANG_OPS
