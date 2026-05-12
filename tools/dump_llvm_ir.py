#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import _avelang_bindings as _C
from avelang.compiler import code_generator as cg
from avelang.runtime.jit import JITCallable


def _load_module_from_path(path: Path):
    """Load a module from a file path."""
    # Execute the user file so @avelang.jit decorators run and create
    # JITCallable objects we can inspect. We intentionally avoid inserting the
    # module into sys.modules to keep the binding module surface minimal.
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module_from_name(module_name: str):
    """Load a module by name (e.g., 'avelang_kernels.amdgpu_gemm')."""
    return importlib.import_module(module_name)


def _resolve_module_target(target: str):
    """
    Resolve a target string to a module and optional function name.

    Supports formats:
    - path/to/file.py:function_name
    - module.submodule:function_name
    - path/to/file.py (function name must be provided separately)
    - module.submodule (function name must be provided separately)
    """
    function_name = None

    if ":" in target:
        module_spec, function_name = target.rsplit(":", 1)
    else:
        module_spec = target

    # Try to load as file path first
    path = Path(module_spec)
    if path.exists() and path.suffix == ".py":
        module = _load_module_from_path(path)
    else:
        # Try to load as module name
        try:
            module = _load_module_from_name(module_spec)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import module '{module_spec}'. Make sure the module is in the Python path."
            ) from e

    return module, function_name


def _find_jit_function(module, name: str):
    if not hasattr(module, name):
        candidates = [key for key, value in vars(module).items() if isinstance(value, JITCallable)]
        raise RuntimeError(f"Function '{name}' not found. JIT callables: {candidates}")
    fn = getattr(module, name)
    if not isinstance(fn, JITCallable):
        raise RuntimeError(f"Attribute '{name}' is not a JIT callable")
    return fn


def _list_jit_functions(module):
    """List all JIT functions in a module."""
    functions = []
    for key, value in vars(module).items():
        if isinstance(value, JITCallable):
            fn = value
            file_name = fn.fn.__code__.co_filename
            line_no = fn.fn.__code__.co_firstlineno
            functions.append(
                {
                    "name": key,
                    "file": file_name,
                    "line": line_no,
                }
            )
    return functions


def _serialize_global_constexprs(global_constants) -> str:
    if not global_constants:
        return "[]"
    constexprs_list = []
    for name, info in global_constants.items():
        constexprs_list.append(
            {
                "name": name,
                "type": info["type"],
                "value": info["value"],
            }
        )
    return json.dumps(constexprs_list)


def _serialize_constexprs_from_jit_fn(jit_fn) -> str:
    """Collect constexprs from a JITCallable for serialization."""
    # For parameter constants (not applicable for kernels with fixed signatures)
    constants = {}

    # For global constants
    collect_globals = getattr(jit_fn, "_collect_global_constexprs", None)
    if callable(collect_globals):
        global_constants = collect_globals()
    else:
        global_constants = {}

    constexprs_list = []
    # Add parameter constants (if any)
    for k, info in constants.items():
        param_name = jit_fn.arg_names[k[0]]
        constexprs_list.append(
            {
                "name": param_name,
                "type": info["type"],
                "value": info["value"],
            }
        )
    # Add global constants
    for name, info in global_constants.items():
        constexprs_list.append(
            {
                "name": name,
                "type": info["type"],
                "value": info["value"],
            }
        )
    return json.dumps(constexprs_list)


def _build_generator(jit_fn, constexprs_json: str):
    jit_deps = cg._collect_jit_dependencies(jit_fn)
    import_module = cg._build_import_module([jit_fn, *jit_deps])

    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(import_module)

    for dep in jit_deps:
        generator.add_jit_dependency(dep.parse())

    for dep in jit_deps:
        dep_func = cg._get_function_def(dep.parse())
        # Collect global constexprs from this dependency
        dep_globals = {}
        collect_globals = getattr(dep, "_collect_global_constexprs", None)
        if callable(collect_globals):
            dep_globals = collect_globals()
        generator.visit_function_def(dep_func, _serialize_global_constexprs(dep_globals), "jit")

    kernel_func = cg._get_function_def(jit_fn.parse())
    # Auto-detect constexprs if not provided manually
    if constexprs_json == "[]":
        constexprs_json = _serialize_constexprs_from_jit_fn(jit_fn)
    generator.visit_function_def(kernel_func, constexprs_json, "kernel")
    return generator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump LLVM IR for a @avelang.jit kernel.",
        epilog=(
            "Examples:\n"
            "  # Dump LLVM IR from a file (specify function separately)\n"
            "  dump_llvm_ir.py path/to/kernel.py my_kernel\n\n"
            "  # Dump LLVM IR using module path syntax\n"
            "  dump_llvm_ir.py avelang_kernels.amdgpu_gemm:_gemm_pipeline_transposed_b_kernel\n\n"
            "  # Dump LLVM IR from file with inline function specification\n"
            "  dump_llvm_ir.py path/to/kernel.py:my_kernel\n\n"
            "  # List all JIT functions in a module\n"
            "  dump_llvm_ir.py --list avelang_kernels.amdgpu_gemm\n"
            "  dump_llvm_ir.py --list path/to/kernel.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        help="Target in format: [path/to/file.py|module.name][:function_name]. "
        "If function_name is not provided, use --function argument.",
    )
    parser.add_argument(
        "function",
        nargs="?",
        help="Kernel function name (can also be specified as target:function)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all JIT functions in the module and exit",
    )
    parser.add_argument(
        "--target-triple",
        default="amdgcn-amd-amdhsa",
        help="Target triple (default: nvptx64-nvidia-cuda for NVIDIA, amdgcn-amd-amdhsa for AMDGPU)",
    )
    parser.add_argument(
        "--target-chipset",
        default="gfx90a",
        help="Target GPU chipset (default: sm_80 for NVIDIA, gfx90a for AMDGPU)",
    )
    parser.add_argument(
        "-O",
        "--opt-level",
        type=int,
        default=2,
        help="Optimization level (0-3)",
    )
    parser.add_argument(
        "--constexprs-json",
        default="[]",
        help="Constexprs JSON (default: [])",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output file")
    args = parser.parse_args()

    # Resolve the target to module and function name
    module, function_from_target = _resolve_module_target(args.target)

    # Determine function name from various sources
    function_name = function_from_target or args.function
    if not function_name and not args.list:
        parser.error(
            "Function name must be provided either as TARGET:FUNCTION or via --function argument, or use --list to see available functions"
        )

    # List mode
    if args.list:
        functions = _list_jit_functions(module)
        if not functions:
            print(f"No JIT functions found in {args.target}")
            return 0

        print(f"Available JIT functions in {args.target}:")
        for fn_info in functions:
            print(f"  - {fn_info['name']} (defined at {fn_info['file']}:{fn_info['line']})")
        return 0

    # Normal mode - dump LLVM IR
    jit_fn = _find_jit_function(module, function_name)

    generator = _build_generator(jit_fn, args.constexprs_json)
    llvm_ir = generator.get_llvm_ir(args.target_triple, args.target_chipset, args.opt_level)

    if args.output:
        args.output.write_text(llvm_ir)
    else:
        print(llvm_ir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
