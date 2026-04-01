from hpc_kers.libtensor import lib, TensorPtr, Error, DTypes, PriOp, c_size_t, POINTER
import ctypes

class Tensor:
    def __init__(self, shape=None, dtype='REAL64', data=None, _ptr=None):
        self._err = Error()
        self._ptr = _ptr
        if _ptr:
            return
        if shape is None:
            raise ValueError("Shape required for new tensor")
        self.shape = tuple(shape)
        self.rank = len(shape)
        self.dtype = getattr(DTypes,dtype)
        shape_arr = (c_size_t*self.rank)(*self.shape)
        if data is None:
            self._ptr = lib.tensor_mem_alloc(self.rank, shape_arr, self.dtype, ctypes.byref(self._err))
        else:
            self._ptr = lib.tensor_from_buffer_copy(data, self.rank, shape_arr, self.dtype, ctypes.byref(self._err))
        self._check_error()

    def _check_error(self):
        if self._err.code != 0:
            msg = self._err.msg.decode() if self._err.msg else ""
            raise RuntimeError(f"Tensor Error {self._err.code}: {msg}")

    def __del__(self):
        if self._ptr:
            lib.tensor_mem_free(self._ptr)
            self._ptr = None

    # ---------------- Metadata ----------------
    @property
    def dtype_(self):
        return lib.tensor_meta_dtype(self._ptr, ctypes.byref(self._err))
    @property
    def size(self):
        return lib.tensor_meta_size(self._ptr, ctypes.byref(self._err))
    @property
    def rank_(self):
        return lib.tensor_meta_rank(self._ptr, ctypes.byref(self._err))
    @property
    def shape_(self):
        ptr = lib.tensor_meta_shape(self._ptr, ctypes.byref(self._err))
        self._check_error()
        return tuple(ptr[i] for i in range(self.rank_))

    # ---------------- Views ----------------
    def reshape(self, new_shape):
        rank = len(new_shape)
        shape_arr = (c_size_t*rank)(*new_shape)
        out = lib.tensor_op_view_reshape(self._ptr, rank, shape_arr, ctypes.byref(self._err))
        self._check_error()
        return Tensor(_ptr=out)
    def permute(self, order):
        order_arr = (c_size_t*len(order))(*order)
        out = lib.tensor_op_view_permute(self._ptr, order_arr, ctypes.byref(self._err))
        self._check_error()
        return Tensor(_ptr=out)
    def slice(self, start, stop, step):
        r = len(start)
        start_arr = (c_size_t*r)(*start)
        stop_arr = (c_size_t*r)(*stop)
        step_arr = (c_size_t*r)(*step)
        out = lib.tensor_op_view_slice(self._ptr, start_arr, stop_arr, step_arr, ctypes.byref(self._err))
        self._check_error()
        return Tensor(_ptr=out)
    def expand(self, new_shape):
        rank = len(new_shape)
        shape_arr = (c_size_t*rank)(*new_shape)
        out = lib.tensor_op_view_expand(self._ptr, rank, shape_arr, ctypes.byref(self._err))
        self._check_error()
        return Tensor(_ptr=out)

    # ---------------- Kernels ----------------
    def ew_kernel(self, kernel_func, *others):
        inputs = (TensorPtr*(len(others)+1))()
        inputs[0] = self._ptr
        for i,t in enumerate(others):
            inputs[i+1] = t._ptr
        out = lib.tensor_op_ew_ker(kernel_func, None, inputs, ctypes.byref(self._err))
        self._check_error()
        return Tensor(_ptr=out)

    def rdc_kernel(self, kernel_func, axis=0):
        out = lib.tensor_op_rdc_ker(kernel_func, self._ptr, axis, ctypes.byref(self._err))
        self._check_error()
        return Tensor(_ptr=out)

    # ---------------- Data ----------------
    @classmethod
    def from_const(cls, val, shape, dtype='REAL64'):
        rank = len(shape)
        shape_arr = (c_size_t*rank)(*shape)
        err = Error()
        ptr = lib.tensor_from_const(val, rank, shape_arr, getattr(DTypes,dtype), ctypes.byref(err))
        if err.code != 0: raise RuntimeError(f"Error {err.code}: {err.msg.decode()}")
        return cls(_ptr=ptr)

    @classmethod
    def from_buffer(cls, buffer, shape, dtype='REAL64'):
        rank = len(shape)
        shape_arr = (c_size_t*rank)(*shape)
        err = Error()
        ptr = lib.tensor_from_buffer_copy(buffer, rank, shape_arr, getattr(DTypes,dtype), ctypes.byref(err))
        if err.code != 0: raise RuntimeError(f"Error {err.code}: {err.msg.decode()}")
        return cls(_ptr=ptr)

    @classmethod
    def from_array_1d(cls, data, size, dtype='REAL64'):
        err = Error()
        ptr = lib.tensor_from_array_1d(data, size, getattr(DTypes,dtype), ctypes.byref(err))
        if err.code != 0: raise RuntimeError(f"Error {err.code}: {err.msg.decode()}")
        return cls(_ptr=ptr)

    def to_buffer(self):
        buf = lib.tensor_to_buffer_copy(self._ptr, ctypes.byref(self._err))
        self._check_error()
        return buf

    # ---------------- Debug ----------------
    def print(self):
        lib.tensor_print(self._ptr, ctypes.byref(self._err))
        self._check_error()
    def print_structure(self):
        lib.tensor_print_structure(self._ptr, ctypes.byref(self._err))
        self._check_error()