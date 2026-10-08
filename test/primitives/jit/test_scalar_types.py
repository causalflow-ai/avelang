from __future__ import annotations

import avelang
import avelang.language as al
import pytest
from avelang.backends.amdgpu.driver import make_launcher as make_hip_launcher
from avelang.backends.nvidia.driver import make_launcher as make_cuda_launcher
from avelang.runtime.jit import _normalize_ty

pytestmark = pytest.mark.no_gpu


@pytest.mark.parametrize(
    "name, expected",
    [("u1", "u1"), ("i32", "i32"), ("u32", "u32"), ("f16", "fp16"), ("bf16", "bf16"), ("f32", "fp32"), ("f64", "fp64")],
)
@pytest.mark.parametrize("prefix", ["", "al.", "avelang.language."])
def test_scalar_annotation_normalization(name, expected, prefix):
    assert _normalize_ty(getattr(al, name)) == expected
    assert _normalize_ty(prefix + name) == expected


@pytest.mark.parametrize("annotation", ["al.f32*", "*al.f32", "const al.f32*"])
def test_pointer_annotation_normalization(annotation):
    expected = "*kfp32" if annotation.startswith("const ") else "*fp32"
    assert _normalize_ty(annotation) == expected


@avelang.jit
def scalar_annotations(x: al.f32, y: al.f64, flag: al.u1):
    pass


def test_postponed_jit_scalar_annotations():
    assert [param.annotation_type for param in scalar_annotations.params] == ["fp32", "fp64", "u1"]


@pytest.mark.parametrize("make_launcher", [make_cuda_launcher, make_hip_launcher], ids=["cuda", "hip"])
@pytest.mark.parametrize("ty, cpp_type, format_code", [("fp32", "float", "f"), ("fp64", "double", "d")])
def test_launcher_scalar_float_types(make_launcher, ty, cpp_type, format_code):
    source = make_launcher({}, {0: ty})
    assert source == make_launcher({}, {0: cpp_type})
    assert f"{cpp_type} arg0" in source
    assert f"{cpp_type} _arg0;" in source
    assert f'PyArg_ParseTuple(args, "iiiiiiKK{format_code}"' in source
