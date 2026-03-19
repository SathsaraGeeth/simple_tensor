from liblinf import *

class Tensor:
    _DTYPE_MAP = {
        "fp64": DTypes.FP64,
        "fp32": DTypes.FP32,
        "int64": DTypes.INT64,
        "int32": DTypes.INT32,
        "int16": DTypes.INT16,
        "int8": DTypes.INT8,
        "uint64": DTypes.UINT64,
        "uint32": DTypes.UINT32,
        "uint16": DTypes.UINT16,
        "uint8": DTypes.UINT8,
    }

    def __init__(self, shape, dtype="fp32"):
        if not isinstance(shape, (list, tuple)):
            raise TypeError("shape must be list/tuple")
        if not all(isinstance(x, int) and x > 0 for x in shape):
            raise ValueError("invalid shape")
        if dtype not in self._DTYPE_MAP:
            raise ValueError("invalid dtype")

        self.rank = len(shape)
        self.shape = tuple(shape)
        self.dtype = self._DTYPE_MAP[dtype]

        c_shape = (c_size_t * self.rank)(*shape)

        self._c_tensor = lib.tensor_mem_alloc(
            c_size_t(self.rank),
            c_shape,
            self.dtype
        )

        if not self._c_tensor:
            raise RuntimeError("allocation failed")

    @classmethod
    def _from_ptr(cls, ptr):
        obj = cls.__new__(cls)
        obj._c_tensor = ptr
        return obj