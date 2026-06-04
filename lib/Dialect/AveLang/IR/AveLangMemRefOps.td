//===- AveLangMemRefOps.td - AveLang memref ops -----------*- tablegen -*-===//
//
// This file defines ave memref operations that mirror the MLIR memref
// dialect but operate on !ave.memref types.
//
//===----------------------------------------------------------------------===//

#ifndef AVE_LANG_MEMREF_OPS
#define AVE_LANG_MEMREF_OPS

// Type constraint for ave memref types.
def AveLang_MemRef : TypeConstraint<
    CPred<"::mlir::isa<::causalflow::avelang::dialect::MemRefType>($_self)">,
    "ave memref type">;

def MemRefLike : TypeConstraint<
    CPred<"::mlir::isa<::causalflow::avelang::dialect::MemRefType, ::mlir::MemRefType>($_self)">,
    "avelang or builtin memref type">;

// Base class for ave memref operations.
class AveLang_MemRefOp<string mnemonic, list<Trait> traits = []> :
    Op<AveLang_Dialect, "memref." # mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AveLang MemRef Operations
//===----------------------------------------------------------------------===//

def AveLangMemRefAllocaOp : AveLang_MemRefOp<"alloca", []> {
    let summary = "Allocate a memref on the stack";
    let description = [{
        The `ave.memref.alloca` operation allocates a memref on the stack.
        It mirrors memref.alloca but uses the !ave.memref type.
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes,
                         OptionalAttr<I64Attr>:$alignment);
    let results = (outs AveLang_MemRef:$result);

    let assemblyFormat = [{
        `(` $dynamicSizes `)` attr-dict `:` type($result)
    }];
}

def AveLangMemRefLoadOp : AveLang_MemRefOp<"load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Load from a memref";
    let description = [{
        The `ave.memref.load` operation loads a value from a memref.
        It mirrors memref.load but uses the !ave.memref type.
    }];

    let arguments = (ins MemRefLike:$memref, Variadic<Index>:$indices);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $memref `[` $indices `]` attr-dict `:` type($memref) `->` type($result)
    }];
}

def AveLangMemRefLoadVecOp : AveLang_MemRefOp<"load_vec", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Load a vector from a memref";
    let description = [{
        The `ave.memref.load_vec` operation loads a vector value from a
        memref. It mirrors vector.load but uses the !ave.memref type.
    }];

    let arguments = (ins MemRefLike:$memref, Variadic<Index>:$indices);
    let results = (outs AnyVectorOfAnyRank:$result);

    let assemblyFormat = [{
        $memref `[` $indices `]` attr-dict `:` type($memref) `->` type($result)
    }];
}

def AveLangMemRefStoreOp : AveLang_MemRefOp<"store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Store to a memref";
    let description = [{
        The `ave.memref.store` operation stores a value into a memref.
        It mirrors memref.store but uses the !ave.memref type.
    }];

    let arguments = (ins AnyType:$value, MemRefLike:$memref,
                         Variadic<Index>:$indices);
    let results = (outs);

    let assemblyFormat = [{
        $value `,` $memref `[` $indices `]` attr-dict `:` type($value) `,`
        type($memref)
    }];
}

def AveLangMemRefViewOp : AveLang_MemRefOp<"view", []> {
    let summary = "Create a typed view into a buffer";
    let description = [{
        The `ave.memref.view` operation creates a memref view into an
        underlying buffer with a byte offset and optional sizes.
    }];

    let arguments = (ins MemRefLike:$source, Index:$byteShift,
                         Variadic<Index>:$sizes);
    let results = (outs AveLang_MemRef:$result);

    let assemblyFormat = [{
        $source `[` $byteShift `]` `[` $sizes `]` attr-dict `:` type($source)
        `->` type($result)
    }];
}

def AveLangMemRefCastOp : AveLang_MemRefOp<"cast", []> {
    let summary = "Cast memref to a different type or layout";
    let description = [{
        The `ave.memref.cast` operation casts a memref to another memref
        type. It is a relaxed version of memref.cast: it allows element type
        and layout changes. When present, the optional layout operand provides
        dynamic dimension/stride values (e.g., from make_layout). If no layout
        is provided, both source and result must have static sizes with the
        same total byte size.
    }];

    let arguments = (ins AnyType:$source, Optional<AnyType>:$layout);
    let results = (outs MemRefLike:$result);

    let builders = [
        OpBuilder<(ins "mlir::Value":$source, "mlir::Type":$resultType)>,
        OpBuilder<(ins "mlir::Value":$source, "mlir::Value":$layout,
                      "mlir::Type":$resultType)>
    ];

    let hasVerifier = 1;

    let assemblyFormat = [{
        $source (`,` $layout^ `:` type($layout))? attr-dict `:` type($source)
        `to` type($result)
    }];
}

def AveLangMemRefExtractAlignedPointerAsIndexOp
    : AveLang_MemRefOp<"extract_aligned_pointer_as_index", [Pure]> {
    let summary = "Extract aligned pointer as index";
    let description = [{
        The `ave.memref.extract_aligned_pointer_as_index` operation
        extracts the aligned pointer of an ave-lang memref as an index.
        It mirrors memref.extract_aligned_pointer_as_index but uses the
        !ave.memref type.
    }];

    let arguments = (ins MemRefLike:$source);
    let results = (outs Index:$aligned_pointer);

    let assemblyFormat = [{
        $source `:` type($source) `->` type(results) attr-dict
    }];
}

def AveLangMemRefSubViewOp : AveLang_MemRefOp<"subview", [
    AttrSizedOperandSegments
]> {
    let summary = "Create a subview of a memref";
    let description = [{
        The `ave.memref.subview` operation creates a view into a memref
        with offsets, sizes, and strides. It mirrors memref.subview but uses the
        !ave.memref type.
    }];

    let arguments = (ins MemRefLike:$source,
                         Variadic<Index>:$offsets,
                         Variadic<Index>:$sizes,
                         Variadic<Index>:$strides);
    let results = (outs AveLang_MemRef:$result);

    let assemblyFormat = [{
        $source `[` $offsets `]` `[` $sizes `]` `[` $strides `]` attr-dict `:`
        type($source) `->` type($result)
    }];
}

#endif // AVE_LANG_MEMREF_OPS
