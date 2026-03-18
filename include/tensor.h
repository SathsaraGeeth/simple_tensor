#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

// Data Types
typedef enum {
    FP64,
    FP32,
    INT64,
    INT32,
    INT16,
    INT8,
    UINT64,
    UINT32,
    UINT16,
    UINT8
} DTypes;

// Tensor Structure
typedef struct {
    void*   data;    // data array ptr
    DTypes  dtype;   // scalar type
    size_t  rank;    // number of dimensions
    size_t* shape;   // array of sizes per axis
    size_t  size;    // total number of elements
    size_t* strides;
} Tensor;

// 1. Memory Management: _mem
Tensor* tensor_mem_alloc(size_t rank, const size_t* shape, DTypes dtype);
void    tensor_mem_free(Tensor* t); 

// 2. Metadata Access: _meta
size_t        tensor_meta_rank(const Tensor* t);
size_t        tensor_meta_size(const Tensor* t);
DTypes        tensor_meta_dtype(const Tensor* t);
const size_t* tensor_meta_shape(const Tensor* t);
const size_t* tensor_meta_strides(const Tensor* t);

// 3. Operations: _op

// 3.1 View Operations: _view
Tensor* tensor_op_view_reshape(Tensor* t, size_t new_rank, const size_t* new_shape);
Tensor* tensor_op_view_broadcast(Tensor* t, size_t new_rank, const size_t* new_shape);
Tensor* tensor_op_view_squeeze(Tensor* t);
Tensor* tensor_op_view_unsqueeze(Tensor* t, size_t axis);
Tensor* tensor_op_view_permute(Tensor* t, const size_t* axes);
Tensor* tensor_op_view_concat(const Tensor** tensors, size_t n, size_t axis);
Tensor* tensor_op_view_flatten(Tensor* t);

// 3.2. Elementwise Operations: _ew
Tensor* tensor_op_ew_add(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_sub(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_mul(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_div(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_pow(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_min(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_max(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_eq(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_neq(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_gt(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_gte(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_lt(const Tensor* a, const Tensor* b);
Tensor* tensor_op_ew_lte(const Tensor* a, const Tensor* b);

// 3.3. Scalar Operations: _sc
Tensor* tensor_op_sc_add(const Tensor* t, double val);
Tensor* tensor_op_sc_sub(const Tensor* t, double val);
Tensor* tensor_op_sc_mul(const Tensor* t, double val);
Tensor* tensor_op_sc_div(const Tensor* t, double val);
Tensor* tensor_op_sc_neg(const Tensor* t);
Tensor* tensor_op_sc_abs(const Tensor* t);

// 3.4. Reduction Operations: _rd
Tensor*  tensor_op_rd_sum(const Tensor* t, size_t axis);
Tensor*  tensor_op_rd_mean(const Tensor* t, size_t axis);
Tensor*  tensor_op_rd_max(const Tensor* t, size_t axis);
Tensor*  tensor_op_rd_min(const Tensor* t, size_t axis);
Tensor*  tensor_op_rd_argmax(const Tensor* t, size_t axis);
Tensor*  tensor_op_rd_argmin(const Tensor* t, size_t axis);

// 3.5. Linear Algebra: _la
Tensor* tensor_op_la_dot(const Tensor* a, const Tensor* b);
Tensor* tensor_op_la_matmul(const Tensor* a, const Tensor* b);
Tensor* tensor_op_la_transpose(const Tensor* t, const size_t* axes);

// 4. Utility Functions: _util
Tensor*     tensor_util_copy(const Tensor* t);
void        tensor_util_print(const Tensor* t);
size_t      tensor_util_index(const Tensor* t, const size_t* indices);
void*       tensor_util_data_ptr(Tensor* t, const size_t* indices);
const void* tensor_util_const_data_ptr(const Tensor* t, const size_t* indices);
size_t      tensor_util_dtype_size(DTypes dtype);
bool        tensor_util_is_contiguous(const Tensor* t);
void        tensor_util_compute_strides(Tensor* t);
size_t      tensor_util_offset(const Tensor* t, const size_t* indices);
Tensor*     tensor_util_like(const Tensor* t);
size_t      tensor_util_numel(const Tensor* t);
void        tensor_util_assert_valid(const Tensor* t);

#endif // TENSOR_H
