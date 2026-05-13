class dtype:
    SINT_TYPES = ["i8", "i16", "i32", "i64"]
    UINT_TYPES = ["u1", "u8", "u16", "u32", "u64"]
    FP_TYPES = ["fp16", "bf16", "fp32", "fp64"]
    STANDARD_FP_TYPES = ["fp16", "bf16", "fp32", "fp64"]
    OTHER_TYPES = ["void"]

    def __init__(self, name):
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name


# scalar types
# Signed integer types
i8 = dtype("i8")
i16 = dtype("i16")
i32 = dtype("i32")
i64 = dtype("i64")

# Unsigned integer types
u1 = dtype("u1")
u8 = dtype("u8")
u16 = dtype("u16")
u32 = dtype("u32")
u64 = dtype("u64")

# Floating-point types
f16 = dtype("fp16")
bf16 = dtype("bf16")
f32 = dtype("fp32")
f64 = dtype("fp64")

# Other types
void = dtype("void")


dynamic = -1


class Tensor:
    """Multi-dimensional tensor type."""

    def __init__(self, shape, element_ty):
        self.shape = shape
        self.element_ty = element_ty


class pointer_type:
    def __init__(self, element_ty):
        self.element_ty = element_ty


Pointer = pointer_type


class constexpr_type:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, constexpr_type) and self.value == other.value

    def __repr__(self) -> str:
        return f"constexpr_type[{self.value}]"


class constexpr:
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        while isinstance(value, constexpr):
            value = value.value
        self.value = value
        self.type = constexpr_type(value)

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __hash__(self):
        return hash((self.value, self.type))


# Built-in functions for GPU kernels
def block_id(dim: int):
    """Get the block ID in the specified dimension."""
    pass


def block_dim(dim: int):
    """Get the block dimension in the specified dimension."""
    pass


def thread_id(dim: int):
    """Get the thread ID in the specified dimension."""
    pass


def grid_dim(dim: int):
    """Get the grid dimension in the specified dimension."""
    pass


def shuffle(value, offset: int, width: int):
    """Shuffle a value across lanes within a subgroup."""
    pass


def shuffle_up(value, offset: int, width: int):
    """Shuffle a value across lanes within a subgroup using upward indexing."""
    pass


def shuffle_down(value, offset: int, width: int):
    """Shuffle a value across lanes within a subgroup using downward indexing."""
    pass


def shuffle_xor(value, offset: int, width: int):
    """Shuffle a value across lanes within a subgroup using XOR lane selection."""
    pass


def min(lhs, rhs):
    """Compute the unsigned minimum of two integer values."""
    pass


def max(lhs, rhs):
    """Compute the unsigned maximum of two integer values."""
    pass


def abs(x):
    """Compute |x| for scalar GPU floating-point values."""
    pass


def exp2(x):
    """Compute 2**x for scalar GPU values."""
    pass


def exp(x):
    """Compute e**x for scalar GPU values."""
    pass


def tanh(x):
    """Compute tanh(x) for scalar GPU values."""
    pass


def log(x):
    """Compute the natural logarithm for scalar GPU values."""
    pass


def log2(x):
    """Compute the base-2 logarithm for scalar GPU values."""
    pass


def erf(x):
    """Compute the error function for scalar GPU values."""
    pass


def sqrt(x):
    """Compute sqrt(x) for scalar GPU values."""
    pass


class _NVVMModule:
    def make_tma_descriptor(self, tensor, smem_layout):
        pass

    def tma_fence(self, desc):
        pass

    def tma_load(
        self, dst, desc, coords, barrier, mbar_id=0, predicate=True,
        multicast_mask=None
    ):
        pass

    def tma_store(self, src, desc, coords, predicate=True):
        pass
    

nvvm = _NVVMModule()
