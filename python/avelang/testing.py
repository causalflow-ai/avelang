def has_cuda():
    import torch

    return torch.cuda.is_available()


def has_cuda_nvidia():
    import torch

    if not torch.cuda.is_available():
        return False
    if torch.version.cuda is None:
        return False
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        return False
    return "nvidia" in name.lower()


def has_rocm():
    import torch

    if not torch.cuda.is_available():
        return False
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def _has_rocm_arch(arch):
    if not has_rocm():
        return False

    from .backends.amdgpu.driver import AmdgpuDriver

    return AmdgpuDriver.get_current_target().chip == arch


def has_gfx950():
    return _has_rocm_arch("gfx950")


def has_gfx942():
    return _has_rocm_arch("gfx942")
