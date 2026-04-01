import os
import ctypes
from ctypes import c_size_t, c_void_p, POINTER, c_int, c_bool, c_char_p

# Load shared library
_lib_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(_lib_dir, "libtensor.so")
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

class PriOp(ctypes.c_int):
    ADD=0;  SUB=1;  MUL=2;  DIV=3;  REM=4
    NEG=5;  ABS=6;  FMA=7;  MIN=8;  MAX=9
    EQ=10;  NEQ=11; GT=12;  GTE=13; LT=14; LTE=15
    AND=16; OR=17;  XOR=18; NOT=19
    SHL=20; SHR=21; SAR=22; CLZ=23; CTZ=24; POPCNT=25


class CTensor(ctypes.Structure):
    pass

TensorPtr = POINTER(CTensor)

# 1. Memory Management
lib.tensor_mem_alloc.restype = TensorPtr
lib.tensor_mem_alloc.argtypes = [c_size_t, POINTER(c_size_t), c_int, POINTER(Error)]

lib.tensor_mem_init.restype = TensorPtr
lib.tensor_mem_init.argtypes = [c_size_t, POINTER(c_size_t), c_int, c_void_p, POINTER(Error)]

lib.tensor_mem_free.restype = None
lib.tensor_mem_free.argtypes = [TensorPtr]

lib.tensor_mem_copy.restype = TensorPtr
lib.tensor_mem_copy.argtypes = [TensorPtr, POINTER(Error)]

# 2. Metadata
lib.tensor_meta_dtype.restype = c_int
lib.tensor_meta_dtype.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_meta_size.restype = c_size_t
lib.tensor_meta_size.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_meta_shape.restype = POINTER(c_size_t)
lib.tensor_meta_shape.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_meta_rank.restype = c_size_t
lib.tensor_meta_rank.argtypes = [TensorPtr, POINTER(Error)]

# 3. Elementwise Kernels
EWKerFunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(c_void_p), c_void_p, c_size_t)

lib.tensor_op_ew_prim.restype = TensorPtr
lib.tensor_op_ew_prim.argtypes = [c_int, TensorPtr, POINTER(TensorPtr), POINTER(Error)]

lib.tensor_op_ew_ker.restype = TensorPtr
lib.tensor_op_ew_ker.argtypes = [EWKerFunc, TensorPtr, POINTER(TensorPtr), POINTER(Error)]

# 3. Reduction Kernels
lib.tensor_op_rdc_prim.restype = TensorPtr
lib.tensor_op_rdc_prim.argtypes = [c_int, TensorPtr, c_size_t, POINTER(Error)]

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

# 5. Data Manipulation
lib.tensor_from_const.restype = TensorPtr
lib.tensor_from_const.argtypes = [c_size_t, c_size_t, POINTER(c_size_t), c_int, POINTER(Error)]

lib.tensor_from_buffer.restype = TensorPtr
lib.tensor_from_buffer.argtypes = [c_void_p, c_size_t, POINTER(c_size_t), c_int, POINTER(Error)]

lib.tensor_from_buffer_copy.restype = TensorPtr
lib.tensor_from_buffer_copy.argtypes = [c_void_p, c_size_t, POINTER(c_size_t), c_int, POINTER(Error)]

lib.tensor_from_array_1d.restype = TensorPtr
lib.tensor_from_array_1d.argtypes = [c_void_p, c_size_t, c_int, POINTER(Error)]

lib.tensor_from_nested.restype = TensorPtr
lib.tensor_from_nested.argtypes = [c_void_p, c_size_t, POINTER(c_size_t), c_int, POINTER(Error)]

lib.tensor_to_buffer.restype = c_void_p
lib.tensor_to_buffer.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_to_buffer_copy.restype = c_void_p
lib.tensor_to_buffer_copy.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_to_array_1d.restype = c_void_p
lib.tensor_to_array_1d.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_to_nested.restype = c_void_p
lib.tensor_to_nested.argtypes = [TensorPtr, POINTER(Error)]

# 6. Debug
lib.tensor_print.restype = None
lib.tensor_print.argtypes = [TensorPtr, POINTER(Error)]

lib.tensor_print_structure.restype = None
lib.tensor_print_structure.argtypes = [TensorPtr, POINTER(Error)]