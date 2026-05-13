import os
import functools
import subprocess
import ast
from pathlib import Path

import torch

from ..compiler import GPUTarget
from ..driver import GPUDriver
from ...runtime.build import compile_module_from_src

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include"), "/usr/local/cuda/include"]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ["cuda"]


@functools.lru_cache()
def libcuda_dirs():
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = "libcuda.so cannot found!\n"
    if locs:
        msg += "Possible files are located at %s." % str(locs)
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += " (requires sudo) to refresh the linker cache."
    assert any(os.path.exists(os.path.join(path, "libcuda.so.1")) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


class CudaUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        self.load_binary = mod.load_binary


LAUNCHER_PROLOGUE = """
#include <cuda.h>
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{
   if (code != CUDA_SUCCESS)
   {
      const char* prefix = "ave-lang Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {0,};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }
}

#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__);

static void ensureCudaContext() {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}


typedef struct _DevicePtrInfo {
    CUdeviceptr dev_ptr;
    bool valid;
} DevicePtrInfo;

typedef struct _TmaCleanupData {
    CUdeviceptr dev_ptr;
} TmaCleanupData;

static void CUDA_CB cleanupTmaDescriptor(void *userData) {
  TmaCleanupData *data = (TmaCleanupData *)userData;
  if (data) {
    if (data->dev_ptr) {
      cuMemFree(data->dev_ptr);
    }
    PyMem_RawFree(data);
  }
}

static PyObject* data_ptr_str = NULL;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    goto cleanup;
  }
  if (!PyLong_Check(ret)) {
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    goto cleanup;
  }
  ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
  if(!ptr_info.dev_ptr)
    return ptr_info;
  uint64_t dev_ptr;
  int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {
      PyErr_Format(PyExc_ValueError,
                   "Pointer argument (at %d) cannot be accessed from ave-lang (CPU tensor?)", idx);
      ptr_info.valid = false;
  } else if (status != CUDA_SUCCESS) {
      CUDA_CHECK(status);  // Catch any other cuda API errors
      ptr_info.valid = false;
  }
  ptr_info.dev_ptr = dev_ptr;
cleanup:
  Py_XDECREF(ret);
  return ptr_info;
}
"""

LAUNCHER_EPILOGUE = """static PyMethodDef ModuleMethods[] = {
  {"launch", PyLaunch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__avelang_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___avelang_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if(data_ptr_str == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


def _literal_int_tuple(node):
    if not isinstance(node, ast.Tuple):
        return None
    values = []
    for elt in node.elts:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, int):
            return None
        values.append(elt.value)
    return tuple(values)


def _is_attr_call(node, name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    )


def _dtype_to_tma_enum(dtype):
    name = getattr(dtype, "name", str(dtype))
    return {
        "u8": "CU_TENSOR_MAP_DATA_TYPE_UINT8",
        "u16": "CU_TENSOR_MAP_DATA_TYPE_UINT16",
        "u32": "CU_TENSOR_MAP_DATA_TYPE_UINT32",
        "i32": "CU_TENSOR_MAP_DATA_TYPE_INT32",
        "u64": "CU_TENSOR_MAP_DATA_TYPE_UINT64",
        "i64": "CU_TENSOR_MAP_DATA_TYPE_INT64",
        "fp16": "CU_TENSOR_MAP_DATA_TYPE_FLOAT16",
        "fp32": "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
        "fp64": "CU_TENSOR_MAP_DATA_TYPE_FLOAT64",
        "bf16": "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
    }.get(name)


def _dtype_size(dtype):
    name = getattr(dtype, "name", str(dtype))
    return {
        "u8": 1,
        "u16": 2,
        "u32": 4,
        "i32": 4,
        "u64": 8,
        "i64": 8,
        "fp16": 2,
        "fp32": 4,
        "fp64": 8,
        "bf16": 2,
    }.get(name)


def _default_strides(shape):
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def collect_tma_descriptor_specs(src):
    arg_names = list(src.fn.signature.parameters.keys())
    arg_indices = {name: idx for idx, name in enumerate(arg_names)}
    layout_dims = {}
    specs = []

    def dims_from_layout(node):
        if isinstance(node, ast.Name):
            return layout_dims.get(node.id)
        if _is_attr_call(node, "make_layout") and node.args:
            return _literal_int_tuple(node.args[0])
        return None

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and _is_attr_call(node.value, "make_layout")
                and node.value.args
            ):
                dims = _literal_int_tuple(node.value.args[0])
                if dims is not None:
                    layout_dims[node.targets[0].id] = dims
            self.generic_visit(node)

        def visit_Call(self, node):
            if _is_attr_call(node, "make_tma_descriptor") and len(node.args) >= 2:
                tensor_node = node.args[0]
                if isinstance(tensor_node, ast.Name) and tensor_node.id in arg_indices:
                    box_dims = dims_from_layout(node.args[1])
                    annotation = src.fn.signature.parameters[tensor_node.id].annotation
                    shape = getattr(annotation, "shape", None)
                    element_ty = getattr(annotation, "element_ty", None)
                    tma_dtype = _dtype_to_tma_enum(element_ty)
                    elem_size = _dtype_size(element_ty)
                    if box_dims and shape and tma_dtype and elem_size:
                        shape = tuple(shape)
                        strides = _default_strides(shape)
                        specs.append(
                            {
                                "arg_index": arg_indices[tensor_node.id],
                                "rank": len(shape),
                                "global_dims": tuple(reversed(shape)),
                                "global_strides": tuple(
                                    stride * elem_size for stride in reversed(strides[:-1])
                                ),
                                "box_dims": tuple(reversed(box_dims)),
                                "dtype": tma_dtype,
                            }
                        )
            self.generic_visit(node)

    Visitor().visit(src.fn.parse())
    return specs


def make_launcher(constants, signature, tma_specs=None) -> str:
    tma_specs = tma_specs or []

    def ty_to_cpp(ty):
        TYPE_MAP = {
            "i32": "int",
            "i64": "long long",
            "float": "float",
            "double": "double",
            "u32": "unsigned",
            "u64": "unsigned long long",
        }
        if ty[0] == "*":
            return "CUdeviceptr"
        elif ty in TYPE_MAP:
            return TYPE_MAP[ty]
        else:
            raise ValueError(f"Unknown type: {ty}")

    # Grid and block dimensions and function
    _BASE_ARGS_FORMAT = "iiiiiiKK"

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ",".join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == "*":
            return "PyObject*"
        if ty in "constexpr":
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        FORMAT_MAP = {
            "int": "i",
            "long long": "L",
            "unsigned": "I",
            "unsigned long long": "K",
            "float": "f",
            "double": "d",
        }
        if ty in "constexpr":
            return "O"
        if ty[0] == "*":
            return "O"
        else:
            return FORMAT_MAP[ty_to_cpp(ty)]

    args_format = "".join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}

    args_list = ", " + ", ".join(f"&_arg{i}" for i, _ in signature.items()) if len(signature) > 0 else ""

    arg_decl_list = []
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        # if ty in FLOAT_STORAGE_TYPE:
        #    arg_decl_list.append(f"{FLOAT_STORAGE_TYPE[ty]} arg{i}")
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")

    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        # elif ty in FLOAT_STORAGE_TYPE:
        #    internal_args_list.append(f"_arg{i}_storage")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]

    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.extend(f"&tma_desc_dev{i}" for i in range(len(tma_specs)))
    if params:
        params_decl = f"    void *params[] = {{\n        {', '.join(params)},\n    }};"
        params_arg = "params"
    else:
        params_decl = "    void **params = NULL;"
        params_arg = "params"

    arg_decl = ", ".join(arg_decl_list)
    extracted_decls = "\n".join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])
    ptr_decls_text = "\n".join(ptr_decls)
    internal_args = ", ".join(internal_args_list)
    launch_args = f", {internal_args}" if internal_args else ""

    tma_decl_list = []
    for i, spec in enumerate(tma_specs):
        rank = spec["rank"]
        global_dims = ", ".join(str(v) for v in spec["global_dims"])
        global_strides = ", ".join(str(v) for v in spec["global_strides"])
        box_dims = ", ".join(str(v) for v in spec["box_dims"])
        element_strides = ", ".join("1" for _ in range(rank))
        strides_arg = f"tma_global_strides{i}" if rank > 1 else "NULL"
        tma_decl_list.append(
            f"""
    CUtensorMap tma_desc_host{i};
    cuuint64_t tma_global_dims{i}[{rank}] = {{{global_dims}}};
    cuuint32_t tma_box_dims{i}[{rank}] = {{{box_dims}}};
    cuuint32_t tma_element_strides{i}[{rank}] = {{{element_strides}}};
    CUdeviceptr tma_desc_dev{i} = 0;
"""
        )
        if rank > 1:
            tma_decl_list.append(
                f"    cuuint64_t tma_global_strides{i}[{rank - 1}] = {{{global_strides}}};\n"
            )
        tma_decl_list.append(
            f"""
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &tma_desc_host{i}, {spec["dtype"]}, {rank},
        (void *)arg{spec["arg_index"]}, tma_global_dims{i}, {strides_arg},
        tma_box_dims{i}, tma_element_strides{i},
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CUDA_CHECK(cuMemAlloc(&tma_desc_dev{i}, sizeof(CUtensorMap)));
    CUDA_CHECK(cuMemcpyHtoDAsync(tma_desc_dev{i}, &tma_desc_host{i},
                                 sizeof(CUtensorMap), stream));
"""
        )
    tma_decls = "".join(tma_decl_list)

    tma_cleanup_list = []
    for i in range(len(tma_specs)):
        tma_cleanup_list.append(
            f"""
    TmaCleanupData *tma_cleanup{i} =
        (TmaCleanupData *)PyMem_RawMalloc(sizeof(TmaCleanupData));
    if (tma_cleanup{i}) {{
      tma_cleanup{i}->dev_ptr = tma_desc_dev{i};
      CUDA_CHECK(cuLaunchHostFunc(stream, cleanupTmaDescriptor, tma_cleanup{i}));
    }}
"""
        )
    tma_cleanups = "".join(tma_cleanup_list)

    cuda_launch = f"""
static void CudaLaunch(int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, CUstream stream, CUfunction func{", " + arg_decl if arg_decl else ""}) {{
{tma_decls}
{params_decl}
    CUDA_CHECK(cuLaunchKernel(func, grid_x, grid_y, grid_z, block_x, block_y, block_z, 0, stream, {params_arg}, NULL));
{tma_cleanups}
}} 

static PyObject *PyLaunch(PyObject *self, PyObject *args) {{
    int grid_x, grid_y, grid_z, block_x, block_y, block_z;
    unsigned long long stream;
    CUfunction func;
    {extracted_decls}

    ensureCudaContext();
    if (!PyArg_ParseTuple(args, "{format}", &grid_x, &grid_y, &grid_z, &block_x, &block_y, &block_z, &stream, &func {args_list})) {{
        return NULL;
    }}
    {ptr_decls_text}

    Py_BEGIN_ALLOW_THREADS;
    CudaLaunch(grid_x, grid_y, grid_z, block_x, block_y, block_z, (CUstream)stream, func{launch_args});
    Py_END_ALLOW_THREADS;

    if (PyErr_Occurred()) {{
      return NULL;
    }}
    Py_RETURN_NONE;
}}
"""
    return LAUNCHER_PROLOGUE + cuda_launch + LAUNCHER_EPILOGUE


class CudaLauncher:
    def __init__(self, src):
        constants = src.constants if hasattr(src, "constants") else dict()

        def arg_idx(x):
            return (src.fn.arg_names.index(x),) if isinstance(x, str) else x

        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tma_specs = collect_tma_descriptor_specs(src)
        src = make_launcher(constants, signature, tma_specs)
        mod = compile_module_from_src(
            src,
            "__avelang_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        self.launch = mod.launch

    def __call__(self, gridX, gridY, gridZ, blockX, blockY, blockZ, stream, function, *args):
        return self.launch(gridX, gridY, gridZ, blockX, blockY, blockZ, stream, function, *args)


class NvidiaDriver(GPUDriver):
    def __init__(self):
        super().__init__()
        self.launcher_cls = CudaLauncher
        self.utils = CudaUtils()

    @classmethod
    def is_active(cls):
        return torch.cuda.is_available()

    @classmethod
    def get_current_target(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device available for NvidiaDriver.")

        major, minor = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name().lower()
        arch = f"sm_{major}{minor}"
        if major == 9 and ("h100" in name or "h800" in name):
            arch = "sm_90a"

        return GPUTarget("nvptx64-nvidia-cuda", arch)
