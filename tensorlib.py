import os
import ctypes
from ctypes import c_size_t, c_void_p, POINTER, c_int, c_bool, c_char_p

# Load shared library
_lib_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(_lib_dir, "libtensor_c.so")
lib = ctypes.CDLL(lib_path)

# Types
c_size_p = POINTER(c_size_t)

# Enum
class DTypes(ctypes.c_int):
    REAL64   = 0
    REAL32   = 1
    INT64  = 2
    INT32  = 3
    INT16  = 4
    INT8   = 5
    UINT64 = 6
    UINT32 = 7
    UINT16 = 8
    UINT8  = 9

class ErrorCode(ctypes.c_int):
    ERR_OK = 0
    ERR_NULL_PTR = 1
    ERR_INVALID_ARG = 2
    ERR_INVALID_SHAPE = 3
    ERR_INVALID_RANK = 4
    ERR_INVALID_AXIS = 5
    ERR_SHAPE_MISMATCH = 6
    ERR_NOT_BROADCASTABLE = 7
    ERR_INVALID_DTYPE = 8
    ERR_DTYPE_MISMATCH = 9
    ERR_MALLOC_FAIL = 10
    ERR_OUT_OF_BOUNDS = 11
    ERR_NOT_IMPLEMENTED = 12
    ERR_UNKNOWN = 13

class Error(ctypes.Structure):
    _fields_ = [
        ("code", c_int),
        ("msg", c_char_p),
    ]

class CTensor(ctypes.Structure):
    pass

TensorPtr = POINTER(CTensor)

# 1. Memory Management
lib.tensor_mem_alloc.restype = TensorPtr
lib.tensor_mem_alloc.argtypes = [c_size_t, c_size_p, c_int, POINTER(Error)]

lib.tensor_mem_init.restype = TensorPtr
lib.tensor_mem_init.argtypes = [c_size_t, c_size_p, c_int, c_void_p, POINTER(Error)]

lib.tensor_mem_view_init.restype = TensorPtr
lib.tensor_mem_view_init.argtypes = [c_size_t, c_size_p, c_int, c_void_p, POINTER(Error)]

lib.tensor_mem_init_const.restype = TensorPtr
lib.tensor_mem_init_const.argtypes = [c_size_t, c_size_p, c_int, c_size_t, POINTER(Error)]

lib.tensor_mem_free.restype = None
lib.tensor_mem_free.argtypes = [TensorPtr]

lib.tensor_mem_copy.restype = TensorPtr
lib.tensor_mem_copy.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_mem_to_array.restype = c_void_p
lib.tensor_mem_to_array.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_mem_view_to_array.restype = c_void_p
lib.tensor_mem_view_to_array.argtypes = [TensorPtr, POINTER(Error)]

# 3. Elementwise
EWKerFunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(c_void_p), c_void_p, c_size_t)

lib.tensor_op_ew_ker.restype = TensorPtr
lib.tensor_op_ew_ker.argtypes = [EWKerFunc, TensorPtr, POINTER(TensorPtr), POINTER(Error)]

# 3. Reduction
lib.tensor_op_rdc_ker.restype = TensorPtr
lib.tensor_op_rdc_ker.argtypes = [EWKerFunc, TensorPtr, c_size_t, POINTER(Error)]

# 3. Views
lib.tensor_op_view_reshape.restype = TensorPtr
lib.tensor_op_view_reshape.argtypes = [TensorPtr, c_size_t, POINTER(c_size_t), POINTER(Error)]

lib.tensor_op_view_permute.restype = TensorPtr
lib.tensor_op_view_permute.argtypes = [TensorPtr, POINTER(c_size_t), POINTER(Error)]

lib.tensor_op_view_slice.restype = TensorPtr
lib.tensor_op_view_slice.argtypes = [TensorPtr, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), POINTER(Error)]

lib.tensor_op_view_expand.restype = TensorPtr
lib.tensor_op_view_expand.argtypes = [TensorPtr, c_size_t, POINTER(c_size_t), POINTER(Error)]

# 4. Utility Functions
lib.tensor__util__dptr.restype = c_void_p
lib.tensor__util__dptr.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor__util__dtype_size.restype = c_size_t
lib.tensor__util__dtype_size.argtypes = [c_int, POINTER(Error)]

lib.tensor__util__is_contiguous.restype = c_bool
lib.tensor__util__is_contiguous.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor__util__offset_from_index.restype = c_size_t
lib.tensor__util__offset_from_index.argtypes = [TensorPtr, POINTER(c_size_t), POINTER(Error)]

lib.tensor__util__shape_equal.restype = c_bool
lib.tensor__util__shape_equal.argtypes = [TensorPtr, TensorPtr, POINTER(Error)]

lib.tensor__util__is_broadcastable.restype = c_bool
lib.tensor__util__is_broadcastable.argtypes = [TensorPtr, TensorPtr, POINTER(Error)]

lib.tensor__util__compute_strides.restype = None
lib.tensor__util__compute_strides.argtypes = [c_size_t, POINTER(c_size_t), POINTER(c_size_t), POINTER(Error)]
