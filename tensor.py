from liblinf import DTypes, lib, LP_CTensor, c_size_p, c_void_p, ctypes

class Tensor:
    def __init__(self, shape, dtype= "fp32"):
        """
        Initialize a Tensor.
        :param shape: a tuple
        :param dtype: a string from {fp64, fp32, int64, int32, int16, int8, uint64, uint32, uint16, uint8}
        """
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be tuple or list, got {type(shape)}")
        if not all(isinstance(x, int) and x > 0 for x in shape):
            raise ValueError(f"All dimensions in shape must be positive integers, got {shape}")
        
        if not isinstance(dtype, str):
            raise TypeError(f"dtype must be string, got {type(dtype)}")
        dtype_lower = dtype.lower()
        if dtype_lower not in _dtype_map:
            raise ValueError(f"Invalid dtype '{dtype}', must be one of {list(_dtype_map.keys())}")

        self._dtype = _dtype_map[dtype_lower]

        # Convert shape to C array
        rank = len(shape)
        c_shape = (ctypes.c_size_t * rank)(*shape)

        # Allocate tensor via C library
        self._c_tensor = lib.tensor_mem_alloc(rank, c_shape, self._dtype)
        if not self._c_tensor:
            raise MemoryError("Failed to allocate tensor")

        # Store Python-friendly shape
        self.shape = tuple(shape)
        self.rank = rank
        self.size = lib.tensor_meta_size(self._c_tensor)

    def __del__(self):
        # Free C tensor memory
        if hasattr(self, "_c_tensor") and self._c_tensor:
            lib.tensor_mem_free(self._c_tensor)
            self._c_tensor = None