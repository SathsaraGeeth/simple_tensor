import os

# import shared library
import ctypes
from   ctypes import c_size_t, c_double, c_void_p, POINTER, c_int
_lib_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(_lib_dir, "libllinf.so")
lib = ctypes.CDLL(lib_path)


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

c_size_p = ctypes.POINTER(ctypes.c_size_t)
c_void_p = ctypes.c_void_p

# Struct
class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", c_void_p),
        ("dtype", ctypes.c_int),
        ("rank", ctypes.c_size_t),
        ("shape", c_size_p),
        ("size", ctypes.c_size_t),
        ("strides", c_size_p),
    ]

LP_CTensor = ctypes.POINTER(CTensor)

# 1. Memory Management: _mem
lib.tensor_mem_alloc.restype = LP_CTensor
lib.tensor_mem_alloc.argtypes = [ctypes.c_size_t, c_size_p, ctypes.c_int]

lib.tensor_mem_free.restype = None
lib.tensor_mem_free.argtypes = [LP_CTensor]

# 2. Metadata Access: _meta
lib.tensor_meta_rank.restype = ctypes.c_size_t
lib.tensor_meta_rank.argtypes = [LP_CTensor]

lib.tensor_meta_size.restype = ctypes.c_size_t
lib.tensor_meta_size.argtypes = [LP_CTensor]

lib.tensor_meta_dtype.restype = ctypes.c_int
lib.tensor_meta_dtype.argtypes = [LP_CTensor]

lib.tensor_meta_shape.restype = c_size_p
lib.tensor_meta_shape.argtypes = [LP_CTensor]

lib.tensor_meta_strides.restype = c_size_p
lib.tensor_meta_strides.argtypes = [LP_CTensor]


# 3. Operations: _op
# 3.1 View Operations: _view
lib.tensor_op_view_reshape.restype = LP_CTensor
lib.tensor_op_view_reshape.argtypes = [LP_CTensor, ctypes.c_size_t, c_size_p]

lib.tensor_op_view_broadcast.restype = LP_CTensor
lib.tensor_op_view_broadcast.argtypes = [LP_CTensor, ctypes.c_size_t, c_size_p]

lib.tensor_op_view_squeeze.restype = LP_CTensor
lib.tensor_op_view_squeeze.argtypes = [LP_CTensor]

lib.tensor_op_view_unsqueeze.restype = LP_CTensor
lib.tensor_op_view_unsqueeze.argtypes = [LP_CTensor, ctypes.c_size_t]

lib.tensor_op_view_permute.restype = LP_CTensor
lib.tensor_op_view_permute.argtypes = [LP_CTensor, c_size_p]

lib.tensor_op_view_concat.restype = LP_CTensor
lib.tensor_op_view_concat.argtypes = [ctypes.POINTER(LP_CTensor), ctypes.c_size_t, ctypes.c_size_t]

lib.tensor_op_view_flatten.restype = LP_CTensor
lib.tensor_op_view_flatten.argtypes = [LP_CTensor]

# 3.2 Elementwise Operations: _ew
for fn in ["add","sub","mul","div","pow","min","max","eq","neq","gt","gte","lt","lte"]:
    f = getattr(lib, f"tensor_op_ew_{fn}")
    f.restype = LP_CTensor
    f.argtypes = [LP_CTensor, LP_CTensor]

# 3.3 Scalar Operations: _sc
for fn in ["add","sub","mul","div","neg","abs"]:
    f = getattr(lib, f"tensor_op_sc_{fn}")
    f.restype = LP_CTensor
    if fn in ["neg","abs"]:
        f.argtypes = [LP_CTensor]
    else:
        f.argtypes = [LP_CTensor, ctypes.c_double]

# 3.4 Reduction Operations: _rd
for fn in ["sum","mean","max","min","argmax","argmin"]:
    f = getattr(lib, f"tensor_op_rd_{fn}")
    f.restype = LP_CTensor
    f.argtypes = [LP_CTensor, ctypes.c_size_t]

# 3.5 Linear Algebra: _la
for fn in ["dot","matmul"]:
    f = getattr(lib, f"tensor_op_la_{fn}")
    f.restype = LP_CTensor
    f.argtypes = [LP_CTensor, LP_CTensor]

lib.tensor_op_la_transpose.restype = LP_CTensor
lib.tensor_op_la_transpose.argtypes = [LP_CTensor, c_size_p]


# 4. Utility Functions: _util
lib.tensor_util_copy.restype = LP_CTensor
lib.tensor_util_copy.argtypes = [LP_CTensor]

lib.tensor_util_print.restype = None
lib.tensor_util_print.argtypes = [LP_CTensor]

lib.tensor_util_index.restype = ctypes.c_size_t
lib.tensor_util_index.argtypes = [LP_CTensor, c_size_p]

lib.tensor_util_data_ptr.restype = c_void_p
lib.tensor_util_data_ptr.argtypes = [LP_CTensor, c_size_p]

lib.tensor_util_const_data_ptr.restype = c_void_p
lib.tensor_util_const_data_ptr.argtypes = [LP_CTensor, c_size_p]

lib.tensor_util_dtype_size.restype = ctypes.c_size_t
lib.tensor_util_dtype_size.argtypes = [ctypes.c_int]

lib.tensor_util_is_contiguous.restype = ctypes.c_bool
lib.tensor_util_is_contiguous.argtypes = [LP_CTensor]

lib.tensor_util_compute_strides.restype = None
lib.tensor_util_compute_strides.argtypes = [LP_CTensor]

lib.tensor_util_offset.restype = ctypes.c_size_t
lib.tensor_util_offset.argtypes = [LP_CTensor, c_size_p]

lib.tensor_util_like.restype = LP_CTensor
lib.tensor_util_like.argtypes = [LP_CTensor]

lib.tensor_util_numel.restype = ctypes.c_size_t
lib.tensor_util_numel.argtypes = [LP_CTensor]

lib.tensor_util_assert_valid.restype = None
lib.tensor_util_assert_valid.argtypes = [LP_CTensor]
