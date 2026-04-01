from tensorlib import lib, TensorPtr, Error, DTypes, c_size_t, EWKerFunc, c_void_p
import ctypes

class Tensor:
    def __init__(self, shape, dtype='REAL32', buffer=None, copconst=None, view=False):
        if shape is None:
            raise ValueError("Shape required for new tensor")
        self._err = Error()
        self.shape = tuple(shape)
        self.rank = len(shape)
        if isinstance(dtype, int):
            self.dtype = dtype
        else:
            self.dtype = getattr(DTypes, dtype)

        shape_arr = (c_size_t * self.rank)(*self.shape)

        if copconst is not None:
            self._ptr = lib.tensor_mem_init_const(self.rank, shape_arr, self.dtype, copconst, ctypes.byref(self._err))
        elif buffer is not None:
            buf_ptr = ctypes.cast(buffer, c_void_p)
            if view:
                self._ptr = lib.tensor_mem_view_init(self.rank, shape_arr, self.dtype, buf_ptr, ctypes.byref(self._err))
            else:
                self._ptr = lib.tensor_mem_init(self.rank, shape_arr, self.dtype, buf_ptr, ctypes.byref(self._err))
        else:
            self._ptr = lib.tensor_mem_alloc(self.rank, shape_arr, self.dtype, ctypes.byref(self._err))

        self._check_error()

    def _check_error(self):
        if self._err.code != 0:
            msg = self._err.msg.decode() if self._err.msg else ""
            raise RuntimeError(f"Tensor Error {self._err.code}: {msg}")

    def deep_copy(self):
        new_ptr = lib.tensor_mem_copy(self._ptr, ctypes.byref(self._err))
        self._check_error()
        return Tensor._from_ptr(new_ptr)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            lib.tensor_mem_free(self._ptr)
            self._ptr = None

    # ---------------- Array conversion ----------------
    def to_array(self):
        data_ptr = lib.tensor_mem_to_array(self._ptr, ctypes.byref(self._err))
        self._check_error()
        ctype = self._ctype_from_dtype()
        numel = self.get_size
        array_type = ctype * numel
        array_ptr = ctypes.cast(data_ptr, ctypes.POINTER(array_type))
        return list(array_ptr.contents)

    def view_to_array(self):
        data_ptr = lib.tensor_mem_view_to_array(self._ptr, ctypes.byref(self._err))
        self._check_error()
        ctype = self._ctype_from_dtype()
        numel = self.get_size
        array_type = ctype * numel
        array_ptr = ctypes.cast(data_ptr, ctypes.POINTER(array_type))
        return list(array_ptr.contents)

    # ---------------- Metadata ----------------
    @property
    def get_dtype(self):
        return lib.tensor_meta_dtype(self._ptr, ctypes.byref(self._err))

    @property
    def get_size(self):
        return lib.tensor_meta_size(self._ptr, ctypes.byref(self._err))

    @property
    def get_rank(self):
        return lib.tensor_meta_rank(self._ptr, ctypes.byref(self._err))

    @property
    def get_shape(self):
        ptr = lib.tensor_meta_shape(self._ptr, ctypes.byref(self._err))
        self._check_error()
        return tuple(ptr[i] for i in range(self.get_rank))

    # ---------------- Views ----------------
    def reshape(self, new_shape):
        rank = len(new_shape)
        shape_arr = (c_size_t * rank)(*new_shape)
        out = lib.tensor_op_view_reshape(self._ptr, rank, shape_arr, ctypes.byref(self._err))
        self._check_error()
        return Tensor._from_ptr(out)

    def permute(self, order):
        order_arr = (c_size_t * len(order))(*order)
        out = lib.tensor_op_view_permute(self._ptr, order_arr, ctypes.byref(self._err))
        self._check_error()
        return Tensor._from_ptr(out)

    def slice(self, start, stop, step):
        r = len(start)
        start_arr = (c_size_t * r)(*start)
        stop_arr = (c_size_t * r)(*stop)
        step_arr = (c_size_t * r)(*step)
        out = lib.tensor_op_view_slice(self._ptr, start_arr, stop_arr, step_arr, ctypes.byref(self._err))
        self._check_error()
        return Tensor._from_ptr(out)

    def expand(self, new_shape):
        rank = len(new_shape)
        shape_arr = (c_size_t * rank)(*new_shape)
        out = lib.tensor_op_view_expand(self._ptr, rank, shape_arr, ctypes.byref(self._err))
        self._check_error()
        return Tensor._from_ptr(out)

    # ---------------- Kernels ----------------
    def ew_kernel(self, kernel_func, *others, out=None):
        inputs = (TensorPtr * (len(others) + 1))()
        inputs[0] = self._ptr
        for i, t in enumerate(others):
            inputs[i + 1] = t._ptr
        if out is None:
            out = Tensor(self.get_shape, dtype=self.dtype)
        lib.tensor_op_ew_ker(EWKerFunc(kernel_func), out._ptr, inputs, ctypes.byref(self._err))
        self._check_error()
        return out

    def rdc_kernel(self, kernel_func, axis=0, out=None):
        ret_ptr = lib.tensor_op_rdc_ker(EWKerFunc(kernel_func), self._ptr, axis, ctypes.byref(self._err))
        self._check_error()
        return Tensor._from_ptr(ret_ptr)

    # ---------------- Derived elementwise ops ----------------
    def _ctype_from_dtype(self):
        mapping = {
            DTypes.REAL32: ctypes.c_float,
            DTypes.REAL64: ctypes.c_double,
            DTypes.INT8:   ctypes.c_int8,
            DTypes.INT16:  ctypes.c_int16,
            DTypes.INT32:  ctypes.c_int32,
            DTypes.INT64:  ctypes.c_int64,
            DTypes.UINT8:  ctypes.c_uint8,
            DTypes.UINT16: ctypes.c_uint16,
            DTypes.UINT32: ctypes.c_uint32,
            DTypes.UINT64: ctypes.c_uint64,
        }
        return mapping[self.dtype]

    def _binary_op(self, other, py_func, out=None):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")
        ctype = self._ctype_from_dtype()
        class CTensor(ctypes.Structure):
            _fields_ = [("data", ctypes.c_void_p)]
        def kernel(inputs, output_ptr, n):
            t1 = ctypes.cast(inputs[0], ctypes.POINTER(CTensor)).contents
            t2 = ctypes.cast(inputs[1], ctypes.POINTER(CTensor)).contents
            a_ptr = ctypes.cast(t1.data, ctypes.POINTER(ctype))
            b_ptr = ctypes.cast(t2.data, ctypes.POINTER(ctype))
            out_ptr = ctypes.cast(output_ptr, ctypes.POINTER(ctype))
            for i in range(n):
                out_ptr[i] = py_func(a_ptr[i], b_ptr[i])
        return self.ew_kernel(kernel, other, out=out)

    def _unary_op(self, py_func, out=None):
        ctype = self._ctype_from_dtype()
        class CTensor(ctypes.Structure):
            _fields_ = [("data", ctypes.c_void_p)]  # minimal needed
        def kernel(inputs, output_ptr, n):
            t = ctypes.cast(inputs[0], ctypes.POINTER(CTensor)).contents
            a_ptr = ctypes.cast(t.data, ctypes.POINTER(ctype))
            out_ptr = ctypes.cast(output_ptr, ctypes.POINTER(ctype))
            for i in range(n):
                out_ptr[i] = py_func(a_ptr[i])
        return self.ew_kernel(kernel, out=out)

    # ---------------- Operators ----------------
    __add__ = lambda self, other: self._binary_op(other, lambda a, b: a + b)
    __sub__ = lambda self, other: self._binary_op(other, lambda a, b: a - b)
    __neg__ = lambda self: self._unary_op(lambda a: -a)
    __mul__ = lambda self, other: self._binary_op(other, lambda a, b: a * b)
    __truediv__ = lambda self, other: self._binary_op(other, lambda a, b: a / b)
    __floordiv__ = lambda self, other: self._binary_op(other, lambda a, b: a // b)
    __mod__ = lambda self, other: self._binary_op(other, lambda a, b: a % b)
    __pow__ = lambda self, other: self._binary_op(other, lambda a, b: a ** b)
    __and__ = lambda self, other: self._binary_op(other, lambda a, b: a & b)
    __or__ = lambda self, other: self._binary_op(other, lambda a, b: a | b)
    __xor__ = lambda self, other: self._binary_op(other, lambda a, b: a ^ b)
    __lshift__ = lambda self, other: self._binary_op(other, lambda a, b: a << b)
    __rshift__ = lambda self, other: self._binary_op(other, lambda a, b: a >> b)
    __invert__ = lambda self: self._unary_op(lambda a: ~a)
    __eq__ = lambda self, other: self._binary_op(other, lambda a, b: a == b)
    __ne__ = lambda self, other: self._binary_op(other, lambda a, b: a != b)
    __lt__ = lambda self, other: self._binary_op(other, lambda a, b: a < b)
    __le__ = lambda self, other: self._binary_op(other, lambda a, b: a <= b)
    __gt__ = lambda self, other: self._binary_op(other, lambda a, b: a > b)
    __ge__ = lambda self, other: self._binary_op(other, lambda a, b: a >= b)

    # ---------------- Representation ----------------
    def __repr__(self):
        return f"Tensor(shape={self.get_shape}, dtype={self.dtype}, rank={self.rank})"

    def __str__(self):
        try:
            arr = self.to_array()
            def build_nested(arr, shape):
                if not shape:
                    return arr[0]
                step = 1
                for s in shape[1:]:
                    step *= s
                return [build_nested(arr[i*step:(i+1)*step], shape[1:]) for i in range(shape[0])]
            return str(build_nested(arr, self.get_shape))
        except Exception as e:
            return f"<Tensor Error: {e}>"
        
    @classmethod
    def _from_ptr(cls, ptr):
        t = object.__new__(cls)
        t._err = Error()
        t._ptr = ptr
        t.shape = tuple(
            lib.tensor_meta_shape(ptr, ctypes.byref(t._err))[i]
            for i in range(lib.tensor_meta_rank(ptr, ctypes.byref(t._err))
        ))
        t.rank = len(t.shape)
        t.dtype = lib.tensor_meta_dtype(ptr, ctypes.byref(t._err))
        return t