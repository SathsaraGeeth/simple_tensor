#include "../include/tensor.h"

#define STUB {if(e){e->code = ERR_NOT_IMPLEMENTED; e->msg = "not implemented";} return NULL;}

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
    t->rank      = rank;
    t->dtype     = dtype;
    t->data      = NULL;
    t->shape     = NULL;
    t->strides   = NULL;
    t->owns_data = true;

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
        tensor__util__compute_strides(rank, t->shape, t->strides, error);
        if (error->code != ERR_OK) goto fail;

        t->size = 1;
        for (size_t i = 0; i < rank; ++i) {
            if (shape[i] == 0 || t->size > SIZE_MAX / shape[i]) {
                error->code = ERR_INVALID_SHAPE;
                error->msg  = "tensor size overflow or zero dimension";
                goto fail;
            }
            t->size *= shape[i];
        }
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
    if (t->owns_data)
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

tensor* tensor_mem_view_init  (size_t rank, const size_t* shape, dtype_t dtype, const void* data, error_t* error){
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
    t->rank      = rank;
    t->dtype     = dtype;
    t->data      = NULL;
    t->shape     = NULL;
    t->strides   = NULL;
    t->owns_data = false;

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
        tensor__util__compute_strides(rank, t->shape, t->strides, error);
        if (error->code != ERR_OK) goto fail;

        t->size = 1;
        for (size_t i = 0; i < rank; ++i) {
            if (shape[i] == 0 || t->size > SIZE_MAX / shape[i]) {
                error->code = ERR_INVALID_SHAPE;
                error->msg  = "tensor size overflow or zero dimension";
                goto fail;
            }
            t->size *= shape[i];
        }
    }

    size_t dtype_size = tensor__util__dtype_size(dtype, error);
    if (error->code != ERR_OK) goto fail;
    t->dtype_size = dtype_size;

    if (t->size > SIZE_MAX / dtype_size) {
        error->code = ERR_MALLOC_FAIL;
        error->msg  = "tensor size overflow";
        goto fail;
    }
    t->data =  (void*)data;

    return t;

// fail cleanup
fail:
    if (t) tensor_mem_free(t);
    return NULL;
}

tensor* tensor_mem_init_const(size_t rank, const size_t* shape, dtype_t dtype, const size_t const_data, error_t* error) {
    if (!error) return NULL;

    error->code = ERR_OK;
    error->msg  = NULL;

    size_t size = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (shape[i] == 0 || size > SIZE_MAX / shape[i]) {
            error->code = ERR_INVALID_SHAPE;
            error->msg  = "invalid shape";
            return NULL;
        }
        size *= shape[i];
    }

    size_t dtype_size = tensor__util__dtype_size(dtype, error);
    if (error->code != ERR_OK) return NULL;

    void* buffer = malloc(size * dtype_size);
    if (!buffer) {
        error->code = ERR_MALLOC_FAIL;
        error->msg  = "buffer allocation failed";
        return NULL;
    }

    for (size_t i = 0; i < size; ++i) {
        switch (dtype) {
            case REAL64: ((double*)buffer)[i] = (double)const_data; break;
            case REAL32: ((float*)buffer)[i]  = (float)const_data;  break;
            case INT64:  ((int64_t*)buffer)[i] = (int64_t)const_data; break;
            case INT32:  ((int32_t*)buffer)[i] = (int32_t)const_data; break;
            case INT16:  ((int16_t*)buffer)[i] = (int16_t)const_data; break;
            case INT8:   ((int8_t*)buffer)[i]  = (int8_t)const_data;  break;
            case UINT64: ((uint64_t*)buffer)[i] = (uint64_t)const_data; break;
            case UINT32: ((uint32_t*)buffer)[i] = (uint32_t)const_data; break;
            case UINT16: ((uint16_t*)buffer)[i] = (uint16_t)const_data; break;
            case UINT8:  ((uint8_t*)buffer)[i]  = (uint8_t)const_data;  break;
            default:
                error->code = ERR_INVALID_DTYPE;
                error->msg  = "unsupported dtype";
                free(buffer);
                return NULL;
        }
    }

    tensor* t = tensor_mem_view_init(rank, shape, dtype, buffer, error);
    if (!t) {
        free(buffer);
        return NULL;
    }
    t->owns_data = true;

    return t;
}

void* tensor_mem_to_array(const tensor* t, error_t* error) {
    if (!error) return NULL;

    error->code = ERR_OK;
    error->msg  = NULL;

    if (!t || !t->data) {
        error->code = ERR_NULL_PTR;
        error->msg  = "tensor or data is NULL";
        return NULL;
    }

    size_t total_bytes = t->size * t->dtype_size;
    void* out_array = malloc(total_bytes);
    if (!out_array) {
        error->code = ERR_MALLOC_FAIL;
        error->msg  = "allocation failed";
        return NULL;
    }

    memcpy(out_array, t->data, total_bytes);
    return out_array;
}

void* tensor_mem_view_to_array(const tensor* t, error_t* error) {
    if (!error) return NULL;

    error->code = ERR_OK;
    error->msg  = NULL;

    if (!t || !t->data) {
        error->code = ERR_NULL_PTR;
        error->msg  = "tensor or data is NULL";
        return NULL;
    }

    return t->data;
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

    for (size_t i = t->rank; i-- > 0; ) {
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
tensor* tensor_op_ew_ker(ew_ker_t kernel, tensor* output, const tensor** inputs, error_t* error){
    if (!error) return NULL;

    error->code = ERR_OK;
    error->msg  = NULL;

    // kernel validation
    if (!kernel) {
        error->code = ERR_NULL_PTR;
        error->msg  = "kernel is NULL";
        return NULL;
    }
    // inputs ptr validation
    if (!inputs) {
        error->code = ERR_NULL_PTR;
        error->msg  = "inputs is NULL";
        return NULL;
    }
    // output validation
    if (!output) {
        error->code = ERR_NULL_PTR;
        error->msg  = "output is NULL";
        return NULL;
    }
    
    kernel((void**)inputs, output->data, output->size);
    return output;
}

/* Reduction */
tensor* tensor_op_rdc_ker(ew_ker_t k,const tensor*t,size_t axis,error_t*e) STUB

/* Views */
tensor* tensor_op_view_reshape(tensor*t,size_t a,const size_t*b,error_t*e) STUB
tensor* tensor_op_view_permute(tensor*t,const size_t*o,error_t*e) STUB
tensor* tensor_op_view_slice(tensor*t,const size_t*a,const size_t*b,const size_t*c,error_t*e) STUB
tensor* tensor_op_view_expand(tensor*t,size_t a,const size_t*b,error_t*e) STUB
