import functools
import os
import subprocess
from pathlib import Path

from ...runtime.build import compile_module_from_src
from ..compiler import GPUTarget
from ..driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))


@functools.lru_cache()
def get_rocm_paths():
    """Discover ROCm installation paths dynamically"""
    try:
        rocm_path = subprocess.check_output(["hipconfig", "--rocmpath"], stderr=subprocess.DEVNULL).decode().strip()
        if os.path.exists(rocm_path):
            return rocm_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    potential_rocm_paths = [
        "/opt/rocm",
    ]

    try:
        for path in os.listdir("/opt"):
            if path.startswith("rocm-"):
                potential_rocm_paths.append(f"/opt/{path}")
    except FileNotFoundError:
        pass

    for path in potential_rocm_paths:
        if os.path.exists(path) and os.path.isdir(f"{path}/bin") and os.path.isdir(f"{path}/include"):
            if os.path.exists(f"{path}/bin/hipcc") or os.path.exists(f"{path}/bin/hipconfig"):
                return path

    return "/opt/rocm"


@functools.lru_cache()
def get_include_dirs():
    """Get HIP include directories"""
    rocm_path = get_rocm_paths()
    include_dirs = [
        os.path.join(dirname, "include"),
        os.path.join(rocm_path, "include"),
        os.path.join(rocm_path, "include", "hip"),
    ]
    return [path for path in include_dirs if os.path.exists(path)]


@functools.lru_cache()
def get_library_dirs():
    """Get HIP library directories"""
    rocm_path = get_rocm_paths()
    common_lib_paths = [
        os.path.join(rocm_path, "lib"),
        os.path.join(rocm_path, "lib64"),
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
    ]

    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path:
        common_lib_paths.extend(env_ld_library_path.split(":"))

    valid_dirs = []
    for path in common_lib_paths:
        if os.path.exists(path) and any(
            os.path.exists(os.path.join(path, lib))
            for lib in ["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"]
        ):
            valid_dirs.append(path)

    return valid_dirs


include_dirs = get_include_dirs()
libdevice_dir = os.path.join(dirname, "lib")
libraries = ["amdhip64", "hsa-runtime64"]


@functools.lru_cache()
def libhip_dirs():
    """Get HIP library directories (deprecated, use get_library_dirs)"""
    return get_library_dirs()


@functools.lru_cache()
def library_dirs():
    """Get all library directories for compilation"""
    return [libdevice_dir, *get_library_dirs()]


class HipUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HipUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="hip_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
            ccflags=["-D__HIP_PLATFORM_AMD__"],
        )
        self.load_binary = mod.load_binary


LAUNCHER_PROLOGUE = """
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <Python.h>
#include <stdbool.h>
#include <cstdint>
#include <string>

static inline void gpuAssert(hipError_t code, const char *file, int line)
{
   if (code != hipSuccess)
   {
      std::string err = "ave-lang Error [HIP]: ";
      err += hipGetErrorString(code);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err.c_str());
      PyGILState_Release(gil_state);
   }
}

#define HIP_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__);

static void ensureHipContext() {
  hipDevice_t device;
  HIP_CHECK(hipGetDevice(&device));
  // HIP context is created automatically when needed
}


typedef struct _DevicePtrInfo {
    hipDeviceptr_t dev_ptr;
    bool valid;
} DevicePtrInfo;

static PyObject* data_ptr_str = NULL;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  PyObject *ret = NULL;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
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
  ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
  if(!ptr_info.dev_ptr)
    goto cleanup;

  // In HIP, device pointers can be used directly
  // No need for additional attribute lookup like in CUDA

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


def make_launcher(constants, signature) -> str:
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
            return "hipDeviceptr_t"
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
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")

    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]

    # Use struct-based parameter passing like the working HIP test
    struct_members = []
    for i, ty in signature.items():
        if ty != "constexpr":
            struct_members.append(f"        {ty_to_cpp(ty)} arg{i};")

    struct_decl = "\n".join(struct_members)

    struct_assignments = []
    for i, ty in signature.items():
        if ty != "constexpr":
            struct_assignments.append(f"    kernel_params.arg{i} = arg{i};")

    struct_assign = "\n".join(struct_assignments)

    arg_decl = ", ".join(arg_decl_list)
    extracted_decls = "\n".join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])
    ptr_decls_text = "\n".join(ptr_decls)
    internal_args = ", ".join(internal_args_list)

    hip_launch = f"""
static void HipLaunch(int grid_x, int grid_y, int grid_z, int block_x, int block_y, int block_z, hipStream_t stream, hipFunction_t func{", " + arg_decl if arg_decl else ""}) {{
    // Create parameter structure compatible with HIP
    struct {{
{struct_decl}
    }} kernel_params;

    // Fill parameter structure
{struct_assign}

    size_t paramSize = sizeof(kernel_params);
    void *config[] = {{
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernel_params,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &paramSize,
        HIP_LAUNCH_PARAM_END
    }};

    // Use hipModuleLaunchKernel with config
    HIP_CHECK(hipModuleLaunchKernel(func, grid_x, grid_y, grid_z, block_x, block_y, block_z, 0, stream, NULL, config));
}}

static PyObject *PyLaunch(PyObject *self, PyObject *args) {{
    int grid_x, grid_y, grid_z, block_x, block_y, block_z;
    unsigned long long stream;
    hipFunction_t func;
    {extracted_decls}

    ensureHipContext();
    if (!PyArg_ParseTuple(args, "{format}", &grid_x, &grid_y, &grid_z, &block_x, &block_y, &block_z, &stream, &func {args_list})) {{
        return NULL;
    }}
    {ptr_decls_text}

    Py_BEGIN_ALLOW_THREADS;
    HipLaunch(grid_x, grid_y, grid_z, block_x, block_y, block_z, (hipStream_t)stream, func{", " + internal_args if internal_args else ""});
    Py_END_ALLOW_THREADS;

    if (PyErr_Occurred()) {{
      return NULL;
    }}
    Py_RETURN_NONE;
}}
"""
    return LAUNCHER_PROLOGUE + hip_launch + LAUNCHER_EPILOGUE


class HipLauncher:
    def __init__(self, src):
        constants = src.constants if hasattr(src, "constants") else dict()

        def arg_idx(x):
            return (src.fn.arg_names.index(x),) if isinstance(x, str) else x

        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, signature)
        mod = compile_module_from_src(
            src,
            "__avelang_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
            ccflags=["-D__HIP_PLATFORM_AMD__"],
            src_extension=".cpp",
        )
        self.launch = mod.launch

    def __call__(self, gridX, gridY, gridZ, blockX, blockY, blockZ, stream, function, *args):
        return self.launch(gridX, gridY, gridZ, blockX, blockY, blockZ, stream, function, *args)


class AmdgpuDriver(GPUDriver):
    def __init__(self):
        super().__init__()
        self.launcher_cls = HipLauncher
        self.utils = HipUtils()

    @classmethod
    def is_active(cls):
        # Check for AMD GPU availability via HIP
        try:
            # First try to detect ROCm/HIP installation
            subprocess.check_output(["hipcc", "--version"], stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: try torch.cuda with ROCm backend
            try:
                import torch

                if not torch.cuda.is_available():
                    return False
                # For ROCm builds, check if hipcc is available
                subprocess.run(["hipcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError, ImportError):
                return False

    @classmethod
    def get_current_target(cls):
        if not cls.is_active():
            raise RuntimeError("No HIP device available for AmdgpuDriver.")

        arch = "gfx90a"

        try:
            result = subprocess.check_output(["rocm-smi", "--showproductname"], stderr=subprocess.DEVNULL).decode()
            lowered = result.lower()
            if "gfx906" in lowered:
                arch = "gfx906"
            elif "gfx908" in lowered:
                arch = "gfx908"
            elif "gfx90a" in lowered:
                arch = "gfx90a"
            if "gfx942" in lowered:
                arch = "gfx942"
            elif "gfx1100" in lowered:
                arch = "gfx1100"
        except subprocess.CalledProcessError:
            pass  # Use default arch if detection fails

        return GPUTarget("amdgcn-amd-amdhsa", arch)
