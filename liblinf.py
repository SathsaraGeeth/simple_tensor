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
    FP64   = 0
    FP32   = 1
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
    ERR_INVALID_SHAPE = 2
    ERR_INVALID_DTYPE = 3
    ERR_NOT_BROADCASTABLE = 4
    ERR_MALLOC_FAIL = 5
    ERR_MISMATCH = 6
    ERR_OUT_OF_BOUNDS = 7
    ERR_NOT_IMPLEMENTED = 8
    ERROR_UNKNOWN = 9

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

LP_Tensor = POINTER(CTensor)

# ---------------- Memory ----------------
lib.tensor_mem_alloc.restype = LP_Tensor
lib.tensor_mem_alloc.argtypes = [c_size_t, c_size_p, c_int, POINTER(Error)]

lib.tensor_mem_init.restype = LP_Tensor
lib.tensor_mem_init.argtypes = [c_size_t, c_size_p, c_int, c_void_p, POINTER(Error)]

lib.tensor_mem_free.restype = None
lib.tensor_mem_free.argtypes = [LP_Tensor]

lib.tensor_mem_copy.restype = LP_Tensor
lib.tensor_mem_copy.argtypes = [LP_Tensor, POINTER(Error)]

# ---------------- Metadata ----------------
lib.tensor_meta_dtype.restype = c_int
lib.tensor_meta_dtype.argtypes = [LP_Tensor, POINTER(Error)]

lib.tensor_meta_size.restype = c_size_t
lib.tensor_meta_size.argtypes = [LP_Tensor, POINTER(Error)]

lib.tensor_meta_shape.restype = c_size_p
lib.tensor_meta_shape.argtypes = [LP_Tensor, POINTER(Error)]

lib.tensor_meta_rank.restype = c_size_t
lib.tensor_meta_rank.argtypes = [LP_Tensor, POINTER(Error)]

# ---------------- Views ----------------
lib.tensor_op_view_reshape.restype = LP_Tensor
lib.tensor_op_view_reshape.argtypes = [LP_Tensor, c_size_t, c_size_p, POINTER(Error)]

lib.tensor_op_view_permute.restype = LP_Tensor
lib.tensor_op_view_permute.argtypes = [LP_Tensor, c_size_p, POINTER(Error)]

lib.tensor_op_view_slice.restype = LP_Tensor
lib.tensor_op_view_slice.argtypes = [LP_Tensor, c_size_p, c_size_p, c_size_p, POINTER(Error)]

lib.tensor_op_view_expand.restype = LP_Tensor
lib.tensor_op_view_expand.argtypes = [LP_Tensor, c_size_t, c_size_p, POINTER(Error)]

# ---------------- Elementwise ----------------
lib.tensor_op_ew_prim.restype = LP_Tensor
lib.tensor_op_ew_prim.argtypes = [c_int, POINTER(LP_Tensor), POINTER(Error)]

lib.tensor_op_ew_ker.restype = LP_Tensor
lib.tensor_op_ew_ker.argtypes = [c_void_p, POINTER(LP_Tensor), POINTER(Error)]

# ---------------- Reduction ----------------
lib.tensor_op_rdc_prim.restype = LP_Tensor
lib.tensor_op_rdc_prim.argtypes = [c_int, LP_Tensor, c_size_t, POINTER(Error)]

lib.tensor_op_rdc_ker.restype = LP_Tensor
lib.tensor_op_rdc_ker.argtypes = [c_void_p, LP_Tensor, c_size_t, POINTER(Error)]

# ---------------- Linear Algebra ----------------
lib.tensor_linalg_matmul.restype = LP_Tensor
lib.tensor_linalg_matmul.argtypes = [LP_Tensor, LP_Tensor, POINTER(Error)]

lib.tensor_linalg_dot.restype = LP_Tensor
lib.tensor_linalg_dot.argtypes = [LP_Tensor, LP_Tensor, POINTER(Error)]

lib.tensor_linalg_mmac.restype = LP_Tensor
lib.tensor_linalg_mmac.argtypes = [LP_Tensor, LP_Tensor, LP_Tensor, ctypes.c_double, ctypes.c_double, POINTER(Error)]

lib.tensor_linalg_transpose.restype = LP_Tensor
lib.tensor_linalg_transpose.argtypes = [LP_Tensor, c_size_p, POINTER(Error)]

# ---------------- Data Manipulation ----------------
lib.tensor_from_const.restype = LP_Tensor
lib.tensor_from_const.argtypes = [c_size_t, c_size_t, c_size_p, c_int, POINTER(Error)]

lib.tensor_from_buffer.restype = LP_Tensor
lib.tensor_from_buffer.argtypes = [c_void_p, c_size_t, c_size_p, c_int, POINTER(Error)]

lib.tensor_from_buffer_copy.restype = LP_Tensor
lib.tensor_from_buffer_copy.argtypes = [c_void_p, c_size_t, c_size_p, c_int, POINTER(Error)]

lib.tensor_from_array_1d.restype = LP_Tensor
lib.tensor_from_array_1d.argtypes = [c_void_p, c_size_t, c_int, POINTER(Error)]

lib.tensor_from_nested.restype = LP_Tensor
lib.tensor_from_nested.argtypes = [c_void_p, c_size_t, c_size_p, c_int, POINTER(Error)]

lib.tensor_to_buffer.restype = c_void_p
lib.tensor_to_buffer.argtypes = [LP_Tensor, POINTER(Error)]

lib.tensor_to_buffer_copy.restype = c_void_p
lib.tensor_to_buffer_copy.argtypes = [LP_Tensor, POINTER(Error)]

lib.tensor_to_array_1d.restype = c_void_p
lib.tensor_to_array_1d.argtypes = [LP_Tensor, POINTER(Error)]

lib.tensor_to_nested.restype = c_void_p
lib.tensor_to_nested.argtypes = [LP_Tensor, POINTER(Error)]

# ---------------- Debug ----------------
lib.tensor_print.restype = None
lib.tensor_print.argtypes = [LP_Tensor, POINTER(Error)]

lib.tensor_print_structure.restype = None
lib.tensor_print_structure.argtypes = [LP_Tensor, POINTER(Error)]
