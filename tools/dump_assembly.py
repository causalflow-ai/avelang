#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
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
        # Check if there's a similar name (e.g., user specified 'gemm_pipeline' but kernel is '_gemm_pipeline_kernel_batch4')
        suggestions = []
        for candidate in candidates:
            if name.replace("_", "").replace("-", "") in candidate.replace("_", "").replace("-", ""):
                suggestions.append(candidate)
            elif candidate.replace("_", "").replace("-", "") in name.replace("_", "").replace("-", ""):
                suggestions.append(candidate)

        error_msg = f"Function '{name}' not found. JIT callables: {candidates}"
        if suggestions:
            error_msg += "\n\nDid you mean one of these?\n  " + "\n  ".join(suggestions)
        raise RuntimeError(error_msg)
    fn = getattr(module, name)
    if not isinstance(fn, JITCallable):
        # Get JIT candidates for suggestions
        candidates = [key for key, value in vars(module).items() if isinstance(value, JITCallable)]
        # Check if there's a similar name
        suggestions = []
        for candidate in candidates:
            if name.replace("_", "").replace("-", "") in candidate.replace("_", "").replace("-", ""):
                suggestions.append(candidate)
            elif candidate.replace("_", "").replace("-", "") in name.replace("_", "").replace("-", ""):
                suggestions.append(candidate)

        error_msg = f"Attribute '{name}' is not a JIT callable (it's a regular Python function)"
        if suggestions:
            error_msg += "\n\nDid you mean one of these JIT functions?\n  " + "\n  ".join(suggestions)
        error_msg += f"\n\nAll JIT callables in module: {candidates}"
        raise RuntimeError(error_msg)
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


def _is_amdgpu_target(target_triple: str) -> bool:
    return "amdgcn" in target_triple or target_triple.startswith("amd")


def _find_symbol_range(binary_path: Path, symbol_name: str) -> tuple[int, int]:
    result = subprocess.run(
        ["llvm-readobj", "--symbols", str(binary_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            "llvm-readobj failed. Ensure it is available in PATH." if not stderr else f"llvm-readobj failed: {stderr}"
        )

    blocks = re.findall(r"Symbol \{\n(.*?)\n  \}", result.stdout, re.DOTALL)
    for block in blocks:
        name_match = re.search(r"^\s*Name:\s+(.+?)\s+\(", block, re.MULTILINE)
        if not name_match or name_match.group(1) != symbol_name:
            continue

        value_match = re.search(r"^\s*Value:\s+0x([0-9A-Fa-f]+)", block, re.MULTILINE)
        size_match = re.search(r"^\s*Size:\s+([0-9]+)", block, re.MULTILINE)
        type_match = re.search(r"^\s*Type:\s+Function", block, re.MULTILINE)
        if not value_match or not size_match or not type_match:
            continue

        start = int(value_match.group(1), 16)
        size = int(size_match.group(1))
        if size <= 0:
            raise RuntimeError(f"Symbol '{symbol_name}' has non-positive size.")
        return start, start + size

    raise RuntimeError(f"Failed to find function symbol '{symbol_name}'.")


def _render_final_binary(
    binary: bytes,
    target_triple: str,
    symbol_name: str | None = None,
) -> str:
    if _is_amdgpu_target(target_triple):
        with tempfile.TemporaryDirectory(prefix="ave-lang-dump-asm-") as tmpdir:
            code_object_path = Path(tmpdir) / "kernel.co"
            code_object_path.write_bytes(binary)

            cmd = [
                "llvm-objdump",
                "-d",
                "--symbolize-operands",
            ]
            if symbol_name:
                start, stop = _find_symbol_range(code_object_path, symbol_name)
                cmd.extend(
                    [
                        f"--start-address=0x{start:x}",
                        f"--stop-address=0x{stop:x}",
                    ]
                )
            cmd.append(str(code_object_path))

            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise RuntimeError(
                    "llvm-objdump failed. Ensure it is available in PATH."
                    if not stderr
                    else f"llvm-objdump failed: {stderr}"
                )
            return result.stdout

    try:
        return binary.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Final compiled artifact for target '{target_triple}' is binary and "
            "this tool does not know how to render it as assembly text."
        ) from exc


def _dump_final_assembly(
    generator,
    target_triple: str,
    target_chipset: str,
    opt_level: int,
    symbol_name: str | None = None,
) -> str:
    binary = generator.compile_to_binary_bytes(target_triple, target_chipset, opt_level)
    return _render_final_binary(binary, target_triple, symbol_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump assembly code for a @avelang.jit kernel.",
        epilog=(
            "Examples:\n"
            "  # Dump assembly from a file (specify function separately)\n"
            "  dump_assembly.py path/to/kernel.py my_kernel\n\n"
            "  # Dump assembly using module path syntax\n"
            "  dump_assembly.py avelang_kernels.amdgpu_gemm:_gemm_pipeline_transposed_b_kernel\n\n"
            "  # Dump assembly from file with inline function specification\n"
            "  dump_assembly.py path/to/kernel.py:my_kernel\n\n"
            "  # List all JIT functions in a module\n"
            "  dump_assembly.py --list avelang_kernels.amdgpu_gemm\n"
            "  dump_assembly.py --list path/to/kernel.py\n"
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
        help="Target triple (default: amdgcn-amd-amdhsa for AMDGPU, nvptx64-nvidia-cuda for NVIDIA)",
    )
    parser.add_argument(
        "--target-chipset",
        default="gfx90a",
        help="Target GPU chipset (default: gfx90a for AMDGPU, sm_80 for NVIDIA)",
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

    # Check for confusing argument: file path as target but module:function as function arg
    if args.function and ":" in args.function:
        path = Path(args.target)
        if path.exists() and path.suffix == ".py":
            print(
                f"Error: You specified both a file path ({args.target}) and a module path ({args.function}).",
                file=sys.stderr,
            )
            print("\nWhen using module:function syntax, only provide the module path:", file=sys.stderr)
            print(f"  dump_assembly.py {args.function} [other options]", file=sys.stderr)
            print("\nOr if the function is in the file:", file=sys.stderr)
            print(f"  dump_assembly.py {args.target}:function_name [other options]", file=sys.stderr)
            return 1

    # Resolve the target to module and function name
    module, function_from_target = _resolve_module_target(args.target)

    # Determine function name from various sources
    function_name = function_from_target or args.function

    # Check if the module has any JIT functions at all
    jit_functions = [key for key, value in vars(module).items() if isinstance(value, JITCallable)]

    # List mode
    if args.list:
        if not jit_functions:
            print(f"No JIT functions found in {args.target}")
            # Try to provide helpful hints
            path = Path(args.target)
            if path.exists() and path.suffix == ".py":
                print("\nHint: This file appears to import JIT functions from other modules.")
                print("Try specifying the module directly, e.g.:")
                print("  dump_assembly.py module.name:function_name")
                print("  dump_assembly.py --list module.name")
            return 0

        print(f"Available JIT functions in {args.target}:")
        for fn_info in _list_jit_functions(module):
            print(f"  - {fn_info['name']} (defined at {fn_info['file']}:{fn_info['line']})")
        return 0

    if not function_name and not args.list:
        parser.error(
            "Function name must be provided either as TARGET:FUNCTION or via --function argument, or use --list to see available functions"
        )

    if not jit_functions:
        path = Path(args.target)
        if path.exists() and path.suffix == ".py":
            raise RuntimeError(
                f"No JIT functions found in {args.target}.\n"
                f"This file appears to import JIT functions from other modules.\n"
                f"Try specifying the module directly, or use --list to see available functions."
            )

    # Normal mode - dump assembly
    jit_fn = _find_jit_function(module, function_name)

    generator = _build_generator(jit_fn, args.constexprs_json)
    assembly = _dump_final_assembly(
        generator,
        args.target_triple,
        args.target_chipset,
        args.opt_level,
        function_name,
    )

    if args.output:
        args.output.write_text(assembly)
    else:
        print(assembly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
