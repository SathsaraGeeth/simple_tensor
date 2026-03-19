#include "../include/tensor.h"

#define STUB {if(e){e->code = ERR_NOT_IMPLEMENTED; e->msg = "not implemented";}}

/* Memory */
tensor* tensor_mem_alloc (size_t rank, const size_t* shape, dtype_t dtype, error_t* error) {
    if (!error) return NULL;

    error->code = ERR_OK;
    error->msg  = NULL;

    // validata inputs
    if (rank > 0 && !shape) {
        error->code = ERR_NULL_PTR;
        error->msg  = "shape is NULL";
        return NULL;
    }

    // memory allocation
    tensor* t = (tensor*)malloc(sizeof(tensor)); 
    if (!t) {
        error->code = ERR_MALLOC_FAIL;
        error->msg  = "tensor allocation failed";
        return NULL;
    }
    // struct assignment
    t->rank    = rank;
    t->dtype   = dtype;
    t->data    = NULL;
    t->shape   = NULL;
    t->strides = NULL;

    if (rank == 0) {            // scalar
        t->size    = 1;
    } else {                    // non-scalar
        t->shape   = (size_t*) malloc(rank * sizeof(size_t));
        t->strides = (size_t*) malloc(rank * sizeof(size_t));

        if (!t->shape || !t->strides) {
            error->code = ERR_MALLOC_FAIL;
            error->msg  = "shape/strides allocation failed";
            goto fail;
        }

        memcpy(t->shape, shape, rank * sizeof(size_t));
        tensor__util__compute_strides(rank, t->shape, t->strides);
        if (error->code != ERR_OK) goto fail;

        t->size = tensor__util__numel(t, error);
        if (error->code != ERR_OK) goto fail;
    }

    size_t dtype_size = tensor__util__dtype_size(dtype, error);
    if (error->code != ERR_OK) goto fail;
    t->dtype_size = dtype_size;

    if (t->size > SIZE_MAX / dtype_size) {
        error->code = ERR_MALLOC_FAIL;
        error->msg  = "tensor size overflow";
        goto fail;
    }
    t->data = malloc(dtype_size * t->size);
    if (!t->data) {
        error->code = ERR_MALLOC_FAIL;
        error->msg  = "data allocation failed";
        goto fail;
    }

    return t;

// fail cleanup
fail:
    if (t) tensor_mem_free(t);
    return NULL;
}

tensor* tensor_mem_init  (size_t rank, const size_t* shape, dtype_t dtype, const void* data, error_t* error) {
    if (!error) return NULL;

    if (!data) {
        error->code = ERR_NULL_PTR;
        error->msg  = "data is NULL";
        return NULL;
    }

    tensor* t = tensor_mem_alloc(rank, shape, dtype, error);
    if (!t) return NULL;

    memcpy(t->data, data, t->dtype_size * t->size);
    return t;
}

void    tensor_mem_free  (tensor* t) {
    if (!t) return;

    free(t->shape);
    free(t->strides);
    free(t->data);
    free(t);
}

tensor* tensor_mem_copy  (const tensor* t, error_t* error){
    if (!error) return NULL;

    error->code = ERR_OK;
    error->msg  = NULL;

    if (!t) {
        error->code = ERR_NULL_PTR;
        error->msg  = "tensor is NULL";
        return NULL;
    }

    tensor* copy = tensor_mem_init(t->rank, t->shape, t->dtype, t->data, error);
    return copy;
}

/* Meta */
dtype_t tensor_meta_dtype(const tensor* t, error_t* error) {
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!t) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return (dtype_t)0;
    }
    return t->dtype;
}
size_t  tensor_meta_size(const tensor* t, error_t* error) {
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!t) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return 0;
    }
    return t->size;
}
size_t* tensor_meta_shape(const tensor* t, error_t* error) {
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!t) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return NULL;
    }
    return t->shape;
}
size_t  tensor_meta_rank(const tensor* t, error_t* error) {
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!t) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return 0;
    }
    return t->rank;
}

/* Utils */
const void*  tensor__util__dptr             (const tensor* t, error_t* error){
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!t) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return NULL;
    }
    return t->data;
}

size_t tensor__util__dtype_size       (dtype_t dtype, error_t* error) {
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    switch (dtype) {
        case REAL64:  return sizeof(double);
        case REAL32:  return sizeof(float);
        case INT64:   return sizeof(int64_t);
        case INT32:   return sizeof(int32_t);
        case INT16:   return sizeof(int16_t);
        case INT8:    return sizeof(int8_t);
        case UINT64:  return sizeof(uint64_t);
        case UINT32:  return sizeof(uint32_t);
        case UINT16:  return sizeof(uint16_t);
        case UINT8:   return sizeof(uint8_t);
        default:
            if (error) {
                error->code = ERR_INVALID_DTYPE;
                error->msg  = "invalid dtype";
            }
            return 0;
    }
}

bool   tensor__util__is_contiguous    (const tensor* t, error_t* error){
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!t) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return false;
    }
    if (t->rank == 0) {
        return true;
    }
    if (!t->shape || !t->strides) {
        if (error) {
            error->code = ERR_INVALID_ARG;
            error->msg  = "tensor shape or strides is NULL";
        }
        return false;
    }

    size_t expected_stride = t->dtype_size;
    if (error->code != ERR_OK) return false;

    for (size_t i = t->rank - 1; i >= 0; --i) {
        if (t->strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= t->shape[i];
    }
    return true;
}

size_t tensor__util__offset_from_index(const tensor* t, const size_t* indices, error_t* error) {
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!t) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return 0;
    }
    if (!indices) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "indices is NULL";
        }
        return 0;
    }
    if (t->rank == 0) {
        return 0;
    }
    if (!t->shape || !t->strides) {
        if (error) {
            error->code = ERR_INVALID_ARG;
            error->msg  = "tensor shape or strides is NULL";
        }
        return 0;
    }

    size_t offset = 0;
    for (size_t i = 0; i < t->rank; ++i) {
        if (indices[i] >= t->shape[i]) {
            if (error) {
                error->code = ERR_OUT_OF_BOUNDS;
                error->msg  = "index out of bounds";
            }
            return 0;
        }
        offset += indices[i] * t->strides[i];
    }
    return offset;
}

bool   tensor__util__shape_equal      (const tensor* a, const tensor* b, error_t* error) {
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!a || !b) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return false;
    }
    if (a->rank != b->rank) {
        return false;
    }
    if (!a->shape || !b->shape) {
        if (error) {
            error->code = ERR_INVALID_ARG;
            error->msg  = "shape is NULL";
        }
        return false;
    }
    for (size_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

bool   tensor__util__is_broadcastable (const tensor* a, const tensor* b, error_t* error) {
    if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }
    if (!a || !b) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "tensor is NULL";
        }
        return false;
    }
    size_t rank_a = a->rank;
    size_t rank_b = b->rank;

    const size_t* shape_a = a->shape;
    const size_t* shape_b = b->shape;

    if (!shape_a || !shape_b) {
        if (error) {
            error->code = ERR_INVALID_ARG;
            error->msg  = "shape is NULL";
        }
        return false;
    }

    // iterate from the back (right-aligned)
    size_t max_rank = (rank_a > rank_b) ? rank_a : rank_b;

    for (size_t i = 0; i < max_rank; i++) {
        size_t dim_a = (i < rank_a) ? shape_a[rank_a - 1 - i] : 1;
        size_t dim_b = (i < rank_b) ? shape_b[rank_b - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}    

void   tensor__util__compute_strides  (size_t rank, const size_t* shape, size_t* strides, error_t* error) {
if (error) {
        error->code = ERR_OK;
        error->msg  = NULL;
    }

    if ((rank > 0) && (!shape || !strides)) {
        if (error) {
            error->code = ERR_NULL_PTR;
            error->msg  = "shape or strides is NULL";
        }
        return;
    }

    if (rank == 0) {
        return;
    }

    strides[rank - 1] = 1;

    for (size_t i = rank - 1; i-- > 0; ) {
        if (shape[i + 1] > 0 && strides[i + 1] > SIZE_MAX / shape[i + 1]) {
            if (error) {
                error->code = ERR_INVALID_SHAPE;
                error->msg  = "stride overflow";
            }
            return;
        }
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}











/* Elementwise */
tensor* tensor_op_ew_prim(pri_op_t op,const tensor**i,error_t*e) STUB
tensor* tensor_op_ew_ker(ew_ker_t k,const tensor**i,error_t*e) STUB

/* Reduction */
tensor* tensor_op_rdc_prim(pri_op_t op,const tensor*t,size_t axis,error_t*e) STUB
tensor* tensor_op_rdc_ker(ew_ker_t k,const tensor*t,size_t axis,error_t*e) STUB

/* Views */
tensor* tensor_op_view_reshape(tensor*t,size_t a,const size_t*b,error_t*e) STUB
tensor* tensor_op_view_permute(tensor*t,const size_t*o,error_t*e) STUB
tensor* tensor_op_view_slice(tensor*t,const size_t*a,const size_t*b,const size_t*c,error_t*e) STUB
tensor* tensor_op_view_expand(tensor*t,size_t a,const size_t*b,error_t*e) STUB



/* Linalg */
tensor* tensor_linalg_matmul(const tensor*a,const tensor*b,error_t*e) STUB
tensor* tensor_linalg_dot(const tensor*a,const tensor*b,error_t*e) STUB
tensor* tensor_linalg_mmac(const tensor*a,const tensor*b,const tensor*c,double x,double y,error_t*e) STUB
tensor* tensor_linalg_transpose(const tensor*t,const size_t*p,error_t*e) STUB
tensor* tensor_linalg_trace(const tensor*t,error_t*e) STUB
tensor* tensor_linalg_diag(const tensor*t,error_t*e) STUB
tensor* tensor_linalg_eye(size_t n,dtype_t d,error_t*e) STUB
tensor* tensor_linalg_norm(const tensor*t,int ord,error_t*e) STUB
tensor* tensor_linalg_solve_linear(const tensor*a,const tensor*b,error_t*e) STUB
tensor* tensor_linalg_inverse(const tensor*t,error_t*e) STUB
tensor* tensor_linalg_cholesky(const tensor*t,error_t*e) STUB
tensor* tensor_linalg_qr(const tensor*t,tensor**q,tensor**r,error_t*e) STUB
tensor* tensor_linalg_lu(const tensor*t,tensor**p,tensor**l,tensor**u,error_t*e) STUB
tensor* tensor_linalg_eig(const tensor*t,tensor**v,error_t*e) STUB
tensor* tensor_linalg_svd(const tensor*t,tensor**u,tensor**s,tensor**vt,error_t*e) STUB

/* Data */
tensor* tensor_from_buffer(void*d,size_t r,const size_t*s,dtype_t dt,error_t*e) STUB
tensor* tensor_from_buffer_copy(const void*d,size_t r,const size_t*s,dtype_t dt,error_t*e) STUB
tensor* tensor_from_array_1d(const void*d,size_t s,dtype_t dt,error_t*e) STUB
tensor* tensor_from_nested(const void*d,size_t r,const size_t*s,dtype_t dt,error_t*e) STUB
void* tensor_to_buffer(const tensor*t,error_t*e){ STUB; return NULL; }
void* tensor_to_buffer_copy(const tensor*t,error_t*e){ STUB; return NULL; }
void* tensor_to_array_1d(const tensor*t,error_t*e){ STUB; return NULL; }
void* tensor_to_nested(const tensor*t,error_t*e){ STUB; return NULL; }

/* Debug */
void tensor_print(const tensor*t,error_t*e) STUB
void tensor_print_structure(const tensor*t,error_t*e) STUB