#include "tensor.h"
#include <stdlib.h>

tensor ERROR_TENSOR = { .error = 1 };

/* Helpers */
static tensor* init_tensor_meta(extent rank, const extent* shape, dtype_t dtype) {
    tensor* t = (tensor*)malloc(sizeof(tensor));
    if (!t) return NULL;

    t->rank          = rank;
    t->dtype         = dtype;
    t->is_contiguous = true;
    t->error         = false;
    t->data          = NULL;
    t->shape         = mem_alloc(rank * sizeof(extent), MEM_HOST);
    t->strides       = mem_alloc(rank * sizeof(extent), MEM_HOST);

    if (!t->shape || !t->strides) {
        mem_free(t->shape);
        mem_free(t->strides);
        free(t);
        return NULL;
    }

    extent* shape_ptr   = (extent*)t->shape->ptr;
    extent* strides_ptr = (extent*)t->strides->ptr;

    t->size = 1;
    for (extent i = 0; i < rank; i++) {
        shape_ptr[i] = shape[i];
        t->size     *= shape[i];
    }

    extent stride = 1;
    for (int i = (int)rank - 1; i >= 0; i--) {
        strides_ptr[i] = stride;
        stride        *= shape_ptr[i];
    }

    return t;
}

/* ----------------- */

tensor* tensor_alloc(extent rank, const extent* shape, dtype_t dtype, mem_loc_t loc) {
    tensor* t = init_tensor_meta(rank, shape, dtype);
    if (!t) return &ERROR_TENSOR;
    t->data = mem_alloc(t->size * dtype_size(dtype), loc);
    if (!t->data) {
        mem_free(t->shape);
        mem_free(t->strides);
        free(t);
        return &ERROR_TENSOR;
    }
    return t;
}

tensor* tensor_view_from(tensor* t, const dptr* data, mem_loc_t loc) {
    if (!t || t->error) return &ERROR_TENSOR;

    if (mem_view(t->data, data, loc) != 0) {
        return &ERROR_TENSOR;
    }

    return t;
}
boolean tensor_free(tensor* t) {
    if (!t || t == &ERROR_TENSOR) return true;

    if (t->data) mem_free(t->data);
    mem_free(t->shape);
    mem_free(t->strides);
    free(t);

    return false;
}

dptr* tensor_view_to(const tensor* t, mem_loc_t loc) {
    if (!t || t->error || !t->data) return NULL;
    if (!t->is_contiguous)          return NULL;
    if (t->data->loc != loc)        return NULL;

    return t->data->ptr;
}
